#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <memory>

#include <onnxruntime_cxx_api.h>

// Helper function for time measurement
inline int64_t get_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// Define model configuration
struct OnnxInception3Config {
    int width = 221;
    int height = 100;
    int channels = 7;
    int num_classes = 3;
};

// ONNX runtime environment
class OnnxRuntime {
public:
    OnnxRuntime() {
        // Create environment
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "inception3-test");
        
        // Configure session options
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Set CPU acceleration
        #ifdef _WIN32
        const char* cpu_provider = "CPUExecutionProvider";
        #else
        const char* cpu_provider = "CPUExecutionProvider";
        #endif
        providers.push_back(cpu_provider);
    }
    
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::vector<const char*> providers;
};

// Inception3 ONNX model
class OnnxInception3Model {
public:
    OnnxInception3Model() : memory_info(Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault)) {
        config.width = 221;
        config.height = 100;
        config.channels = 7;
        config.num_classes = 3;
    }
    
    ~OnnxInception3Model() {
        // Auto cleanup
    }
    
    bool load_model(const std::string& model_path) {
        try {
            printf("Loading ONNX model: %s\n", model_path.c_str());
            
            // Create session
            session = Ort::Session(runtime.env, model_path.c_str(), runtime.session_options);
            
            // Get model input/output information
            Ort::AllocatorWithDefaultOptions allocator;
            
            // Get input names
            size_t num_input_nodes = session.GetInputCount();
            input_names.resize(num_input_nodes);
            
            for (size_t i = 0; i < num_input_nodes; i++) {
                // Replace with GetInputNameAllocated method
                auto input_name = session.GetInputNameAllocated(i, allocator);
                input_names[i] = input_name.get();
                
                // Get input shape
                Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                input_shapes.push_back(tensor_info.GetShape());
                
                printf("  input  #%zu: name=%s, shape=[", i, input_names[i].c_str());
                for (size_t j = 0; j < input_shapes[i].size(); j++) {
                    printf("%lld%s", (long long)input_shapes[i][j], j < input_shapes[i].size() - 1 ? ", " : "");
                }
                printf("]\n");
            }
            
            // Get output names
            size_t num_output_nodes = session.GetOutputCount();
            output_names.resize(num_output_nodes);
            
            for (size_t i = 0; i < num_output_nodes; i++) {
                // Replace with GetOutputNameAllocated method
                auto output_name = session.GetOutputNameAllocated(i, allocator);
                output_names[i] = output_name.get();
                
                // Get output shape
                Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                output_shapes.push_back(tensor_info.GetShape());
                
                printf("  output #%zu: name=%s, shape=[", i, output_names[i].c_str());
                for (size_t j = 0; j < output_shapes[i].size(); j++) {
                    printf("%lld%s", (long long)output_shapes[i][j], j < output_shapes[i].size() - 1 ? ", " : "");
                }
                printf("]\n");
            }
            
            return true;
        } catch (const Ort::Exception& e) {
            fprintf(stderr, "ONNX Runtime error: %s\n", e.what());
            return false;
        }
    }
    
    // Run model inference
    bool run_inference(const std::vector<float>& input_data, std::vector<float>& output_data, int batch_size = 1) {
        try {
            // Prepare input tensor
            std::vector<int64_t> input_dims = {batch_size, config.height, config.width, config.channels};
            
            // Create input tensor
            std::vector<Ort::Value> input_tensors;
            input_tensors.push_back(
                Ort::Value::CreateTensor<float>(
                    memory_info, const_cast<float*>(input_data.data()), 
                    input_data.size(), input_dims.data(), input_dims.size()));
            
            // Prepare input/output names
            std::vector<const char*> input_names_char(input_names.size());
            std::vector<const char*> output_names_char(output_names.size());
            
            for (size_t i = 0; i < input_names.size(); i++) {
                input_names_char[i] = input_names[i].c_str();
            }
            
            for (size_t i = 0; i < output_names.size(); i++) {
                output_names_char[i] = output_names[i].c_str();
            }
            
            // Run inference
            auto output_tensors = session.Run(
                Ort::RunOptions{nullptr}, 
                input_names_char.data(), 
                input_tensors.data(), 
                input_tensors.size(), 
                output_names_char.data(), 
                output_names_char.size());
            
            // Get output data
            const float* output_tensor_data = output_tensors[0].GetTensorData<float>();
            size_t output_tensor_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
            
            output_data.resize(output_tensor_size);
            std::memcpy(output_data.data(), output_tensor_data, output_tensor_size * sizeof(float));
            
            return true;
        } catch (const Ort::Exception& e) {
            fprintf(stderr, "inference error: %s\n", e.what());
            return false;
        }
    }
    
    // Create random input data
    std::vector<float> create_random_input(int batch_size = 1, int seed = 42) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        
        size_t input_size = batch_size * config.channels * config.height * config.width;
        std::vector<float> input_data(input_size);
        
        for (size_t i = 0; i < input_size; i++) {
            input_data[i] = dist(rng);
        }
        
        printf("Random input data created: size=%zu, shape=[%d, %d, %d, %d]\n",
               input_size, batch_size, config.channels, config.height, config.width);
        
        return input_data;
    }
    
    // Create fixed value input data
    std::vector<float> create_fixed_input(float value, int batch_size = 1) {
        size_t input_size = batch_size * config.channels * config.height * config.width;
        std::vector<float> input_data(input_size, value);
        
        printf("Input data created with fixed value: %f, size=%zu, shape=[%d, %d, %d, %d]\n",
               value, input_size, batch_size, config.channels, config.height, config.width);
        
        return input_data;
    }
    
private:
    OnnxRuntime runtime;
    Ort::Session session{nullptr};
    Ort::MemoryInfo memory_info;
    
    OnnxInception3Config config;
    
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::vector<int64_t>> output_shapes;
};

// Test ONNX model
void test_onnx_inception3(OnnxInception3Model& model, int batch_size, int loop_count) {
    // Output vector
    std::vector<float> output_data;
    
    // Run multiple inferences
    for (int i = 0; i < loop_count; i++) {
        printf("Loop %d:\t", i);
        
        // Create fixed value input
        std::vector<float> input_data = model.create_fixed_input(0.1f * i, batch_size);
        
        // Record start time
        int64_t start_time = get_time_ms();
        
        // Run inference
        if (!model.run_inference(input_data, output_data, batch_size)) {
            fprintf(stderr, "Inference failed\n");
            return;
        }
        
        // Record end time
        int64_t end_time = get_time_ms();
        int64_t duration = end_time - start_time;
        
        // Print output results (only first three values, corresponding to the probabilities of three classes)
        if (output_data.size() >= 3) {
            printf("[%1.6f %1.6f %1.6f] cost: %lld ms\n",
                   output_data[0], output_data[1], output_data[2], (long long)duration);
        } else {
            printf("Output data size is less than expected: %zu\n", output_data.size());
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model-path> [batch-size] [loop-count]\n", argv[0]);
        return 1;
    }
    
    // Parse command line arguments
    const char* model_path = argv[1];
    int batch_size = (argc > 2) ? atoi(argv[2]) : 1;
    int loop_count = (argc > 3) ? atoi(argv[3]) : 1;
    
    // printf("ONNX path: %s\n", model_path);
    // printf("batch size: %d\n", batch_size);
    // printf("loop count: %d\n", loop_count);
    
    // Create model instance
    OnnxInception3Model model;
    
    // Load model
    if (!model.load_model(model_path)) {
        fprintf(stderr, "Loading model failed: %s\n", model_path);
        return 1;
    }
    
    // Test model
    test_onnx_inception3(model, batch_size, loop_count);
    
    printf("Test completed successfully\n");
    
    return 0;
}
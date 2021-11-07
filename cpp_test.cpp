#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <assert.h>
#include <chrono>

#define USE_CPU // Chnage USE_CPU to USE_CUDA

#ifdef USE_CUDA
#include "cuda_provider_factory.h"
#endif  // CUDA GPU Enabled

// g++ a.cpp -o a /new_disk1/ninglu_shao/onnxruntime-linux-x64-1.1.2/lib/libonnxruntime.so.1.1.2 -I /new_disk1/ninglu_shao/onnxruntime-linux-x64-1.1.2/include/ -std=c++11
// g++ a.cpp -o a /new_disk1/ninglu_shao/onnxruntime-linux-x64-gpu-1.1.2/lib/libonnxruntime.so.1.1.2 -I /new_disk1/ninglu_shao/onnxruntime-linux-x64-gpu-1.1.2/include/ -std=c++11
// export LD_LIBRARY_PATH=/new_disk1/ninglu_shao/onnxruntime-linux-x64-1.1.2/lib:$LD_LIBRARY_PATH
// export LD_LIBRARY_PATH=/new_disk1/ninglu_shao/onnxruntime-linux-x64-gpu-1.1.2/lib:$LD_LIBRARY_PATH

int main() {
    int round = 1000;
    std::cout << round << std::endl;
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    // session_options.SetIntraOpNumThreads(1);
    // session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#ifdef USE_CUDA
	Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
#endif  // CUDA GPU Enabled

    const char* model_path = "../onnx/a.onnx";
    Ort::Session session(env, model_path, session_options);

    std::vector<const char*> input_node_names = {"input_ids", "token_type_ids", "attention_mask"};
    std::vector<const char*> output_node_names = {"output_0", "output_1"};

    
    // input_ids
    std::vector<int64_t> input_ids_dims = {1, 256};
    size_t input_ids_size = 1 * 256; 
    auto memory_info_1 = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // std::vector<long> input_ids_value = {101, 1037, 3899, 2003, 2770, 2006, 3509,  102};
    // std::vector<long> input_ids_value = {101 ,1037 ,3899 ,2003 ,2770 ,2006 ,3509 ,102 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0};
    std::vector<long> input_ids_value = {101 ,5009 ,4179 ,2192 ,19548 ,2740 ,2729 ,4292 ,12832 ,9871 ,8985 ,2491 ,3218 ,7319 ,2837 ,7632 ,21906 ,6305 ,16994 ,23957 ,8909 ,3736 ,2192 ,19548 ,4708 ,2486 ,4810 ,2198 ,2879 ,3401 ,9108 ,2106 ,3771 ,15091 ,3388 ,9108 ,2902 ,3002 ,12551 ,2047 ,4033 ,6117 ,2118 ,9810 ,9810 ,5288 ,3430 ,3189 ,21754 ,2120 ,2415 ,16514 ,4295 ,2508 ,8099 ,9108 ,2472 ,2407 ,9871 ,3737 ,4712 ,3889 ,9168 ,9108 ,3772 ,2472 ,12654 ,5009 ,4179 ,2192 ,19548 ,2740 ,2729 ,4292 ,2740 ,2729 ,7309 ,16731 ,2860 ,3319 ,2951 ,4953 ,2192 ,28556 ,2192 ,3424 ,3366 ,4523 ,2072 ,2740 ,2729 ,4292 ,2804 ,3563 ,12832 ,5326 ,5335 ,2192 ,19548 ,3218 ,5547 ,6726 ,26835 ,2594 ,12702 ,21759 ,7088 ,6491 ,5776 ,5073 ,2740 ,2729 ,4292 ,3189 ,3319 ,2817 ,10172 ,26629 ,5009 ,4179 ,18661 ,1046 ,2015 ,6904 ,6299 ,2080 ,26629 ,5009 ,4179 ,2192 ,28556 ,2902 ,4483 ,2491 ,1999 ,25969 ,2491 ,23957 ,5009 ,4179 ,21213 ,3449 ,23957 ,11594 ,2837 ,23957 ,5009 ,4179 ,2192 ,28556 ,2192 ,3424 ,3366 ,4523 ,2072 ,2740 ,2729 ,4292 ,1999 ,25969 ,2491 ,3277 ,5995 ,3319 ,2192 ,19548 ,3218 ,16731 ,2860 ,2504 ,29235 ,5073 ,16755 ,2192 ,28556 ,3218 ,5387 ,15316 ,12473 ,29235 ,2047 ,2817 ,24269 ,21150 ,6544 ,2918 ,2192 ,14548 ,2659 ,18949 ,4315 ,18900 ,13706 ,5482 ,2224 ,3319 ,3522 ,2817 ,10580 ,3643 ,4800 ,10521 ,6895 ,28296 ,5649 ,2192 ,19548 ,4712 ,2565 ,4022 ,2535 ,6544 ,2918 ,2192 ,14548 ,5335 ,2192 ,19548 ,3218 ,7680 ,7849 ,4697 ,12832 ,7175 ,3141 ,3277 ,1041 ,2224 ,11707 ,2192 ,3424 ,3366 ,20746 ,2192 ,2843 ,3258 ,6949 ,4147 ,7976 ,4344 ,25464 ,2112 ,3319 ,4045 ,2951 ,4953 ,2192 ,19548 ,5009 ,4179 ,2192 ,19548 ,2740 ,2729 ,4292 ,102};
    Ort::Value input_ids = Ort::Value::CreateTensor<long>(memory_info_1, input_ids_value.data(), input_ids_size, input_ids_dims.data(), 2);
    assert(input_ids.IsTensor());
    // token_type_ids
    std::vector<int64_t> token_type_ids_dims = {1, 256};
    size_t token_type_ids_size = 1 * 256; 
    auto memory_info_2 = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // std::vector<long> token_type_ids_value = {0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<long> token_type_ids_value;
    for (int i = 0; i < 256; ++ i) {
        token_type_ids_value.push_back(0);
    }
    Ort::Value token_type_ids = Ort::Value::CreateTensor<long>(memory_info_2, token_type_ids_value.data(), token_type_ids_size, token_type_ids_dims.data(), 2);
    assert(token_type_ids.IsTensor());
    // attention_mask
    std::vector<int64_t> attention_mask_dims = {1, 256};
    size_t attention_mask_size = 1 * 256; 
    auto memory_info_3 = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // std::vector<long> attention_mask_value = {1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<long> attention_mask_value;
    for (int i = 0; i < 256; ++ i) {
        attention_mask_value.push_back(1);
    }
    Ort::Value attention_mask = Ort::Value::CreateTensor<long>(memory_info_3, attention_mask_value.data(), attention_mask_size, attention_mask_dims.data(), 2);
    assert(attention_mask.IsTensor());

    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(std::move(input_ids));
    ort_inputs.push_back(std::move(token_type_ids));
    ort_inputs.push_back(std::move(attention_mask));

    // test time
    auto begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < round; ++ i) {
        session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(), ort_inputs.size(), output_node_names.data(), 2);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    printf("time cost: %.3f seconds\n", elapsed.count() * 1e-9);
    // auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(), ort_inputs.size(), output_node_names.data(), 2);
  
    // Get pointer to output tensor float values
    // auto type_info = output_tensors[1].GetTensorTypeAndShapeInfo();
    // for (auto x: type_info.GetShape())
    //     std::cout << "shape " << x << std::endl;
    // std::cout << "len " << type_info.GetElementCount() << std::endl;
    // float* sequence = output_tensors[0].GetTensorMutableData<float>();
    // float* pooled = output_tensors[1].GetTensorMutableData<float>();
    // for (size_t i = 0; i != type_info.GetElementCount(); ++ i) {
    //     std::cout << pooled[i] << " ";
    // }
    // std::cout << pooled[0] << std::endl;
    

    return 0;
}
#include <torch/torch.h>
#include <ATen/record_function.h>
//#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/extension.h>
#include <iostream>
#include <vector>
#include "ext_tpp.h"
#include "init.h"
#include "timing.h"
#include "xsmm_functors.h"

using namespace tpp;
using namespace torch::autograd;

#include "tensor_helper.h" 

#define PRINT_T_SIZE(x) std::cout << #x << ": " << x.sizes() << std::endl
#define PRINT_T(x) std::cout << #x << ": " << x << std::endl

//inputs: x1, w1, b1, w2, b2
//return: save_bwd - x1, w1, y1, x2, w2, y2
static std::vector<at::Tensor> cpp_forward(bool bias, int nLayers, std::vector<at::Tensor> inputs) {
        std::vector<at::Tensor> save_bwd;
        at::Tensor t_in = inputs[0];
        int wb_i = 1; // weight bias index
        for(int i=0; i < nLayers; i++) {
            auto t_wt = inputs[wb_i++];// w_i
            auto t_bias = inputs[wb_i++]; //b_i
            auto mm = torch::addmm(t_bias, t_in, t_wt.t());
            auto act = torch::sigmoid(mm);
            save_bwd.push_back(t_in); //in, wt, act
            save_bwd.push_back(t_wt);
            save_bwd.push_back(act);
            t_in = act;
        }
        return save_bwd;
}

// saved_tensors: x1, w1, y1, x2, w2, y2 from forward pass
// grad_act: from Autograd
// return: ret_list - grad_in0, grad_wt1, grad_bias1, grad_wt0, grad_bias0
static std::vector<at::Tensor> cpp_backward(int nLayers, std::vector<at::Tensor> saved_tensors, at::Tensor grad_act) {
    std::vector<at::Tensor> ret_list; //grad_in0, grad_wt1, grad_bias1, grad_wt0, grad_bias0
    ret_list.push_back(grad_act); //dummy push

    int wb_i = 3*nLayers-3; // wb_i initialized to index of in2; -- in1, wt1, act1, in2, wt2, act2
    for(int i=nLayers-1; i >=0; i--) {
        auto dy = at::sigmoid_backward(grad_act, saved_tensors[wb_i+2]); //act[i]
        auto grad_in = torch::mm(dy, saved_tensors[wb_i+1]); //wt[i]
        auto grad_wt = torch::mm(dy.t(), saved_tensors[wb_i]);//in[i]);
        auto grad_bias = dy.sum(0, true);
        if (i==0) {
            ret_list[0] = grad_in;
        }
        ret_list.push_back(grad_wt);
        ret_list.push_back(grad_bias);

        grad_act = grad_in;  
        wb_i -= 3; 
    } 
    return ret_list;
}

REGISTER_SUBMODULE(_mlp_cpp, m) {
    m.def("forward", &cpp_forward, "MLP Forward");
    m.def("backward", &cpp_backward, "MLP Backward");
}


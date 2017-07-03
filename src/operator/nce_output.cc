/*!
 * Copyright (c) 2017 by Contributors
 * \file nceoutput.cc
 * \brief
 * \author weixing.mei
*/

#include "nce_output-inl.h"

namespace mxnet {
namespace op {
    DMLC_REGISTER_PARAMETER(NceOutputParam);

    MXNET_REGISTER_OP_PROPERTY(NceOutput, NceOutputProp)
        .describe(R"code(nce loss compte node 
            document here
        )code" ADD_FILELINE)
        .add_argument("data", "NDArray-or-Symbol", "The input array")
        .add_argument("label", "NDArray-or-Symbol", "The input label")
        .add_argument("label_weight", "NDArray-or-Symbol", "Weight for each label")
        .add_argument("negdis", "NDArray-or-Symbol", "negative sample distribution probability")
        .add_arguments(NceOutputParam::__FIELDS__());

    template <>
    Operator *CreateOp<cpu>(NceOutputParam param, int dtype) {
        Operator *op = NULL;
        MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
            op = new NceOutputOp<cpu, DType>(param);
        });
        return op;
    }

    Operator* NceOutputProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
            std::vector<int> *in_type) const {
        std::vector<TShape> out_shape, aux_shape;
        std::vector<int> out_type, aux_type;
        CHECK(InferType(in_type, &out_type, &aux_type));
        CHECK(InferShape(in_shape, &out_shape, &aux_shape));
        DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
    }
} // namesapce op
} // namespace mxnet

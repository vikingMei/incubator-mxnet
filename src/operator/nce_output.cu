
#include "nce_output-inl.h"

namespace mxnet {
namespace op {
    template<>
    Operator *CreateOp<gpu>(NceOutputParam param, int dtype) {
        Operator *op = NULL;
        MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
            op = new NceOutputOp<gpu, DType>(param);
        });
        return op;
    }

} // namesapce op
} // namespace mxnet

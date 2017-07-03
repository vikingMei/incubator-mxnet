#ifndef MXNET_OPERATOR_NCE_OUTPUT_INL_H_
#define MXNET_OPERATOR_NCE_OUTPUT_INL_H_

#include <vector>

#include <dmlc/parameter.h>
#include <dmlc/logging.h>

#include "mxnet_op.h"
#include "operator_common.h"
#include "elemwise_op_common.h"


namespace mxnet {
namespace op {
    template<typename xpu, typename DType>
    class NceOutputOp;

    struct NceOutputParam;

    template<typename xpu>
    Operator *CreateOp(NceOutputParam param, int dtype);

    namespace NceOutputIdx {
        enum NceOutputOpInputs {kData, kLabel, kWeight, kNegdis, kLnz};
        enum NceOutputOpOutputs {kOut};
        enum NceOutputOpResource {kTempSpace};
    }  // namespace nceoutput


    struct normexp {
        template<typename DType>
        MSHADOW_XINLINE static void Map(int i, DType* out_data, const DType* in_data, const DType* pn, 
                                        const float& lnz, const int& numlab) {
            int k = numlab-1;

            DType y = std::exp(in_data[i]-lnz);
            DType kpn = k*pn[i]; 
            DType norm = y+kpn;

            if(0==i%numlab) {
                out_data[i] = y/norm;
            } else {
                out_data[i] = kpn/norm;
            }
        }
    };


    struct nceoutGrad {
        template<typename DType>
        MSHADOW_XINLINE static void Map(int i, DType* out, DType* y, const DType* labwgt, const int& numlab) {
            // mask out padding label
            DType wgt = labwgt[i];
            if(wgt<1.0e-3) {
                out[i] = 0;
                return;
            }

            if(0==i%numlab) {
                out[i] = y[i]-1;
            } else {
                out[i] = 1-y[i]; 
            }
        }
    };


    struct NceOutputParam: public dmlc::Parameter<NceOutputParam> {
        int dtype;
        float lnz;

        DMLC_DECLARE_PARAMETER(NceOutputParam) {
            DMLC_DECLARE_FIELD(dtype).set_default(mshadow::kFloat32)
                .add_enum("float32", mshadow::kFloat32)
                .add_enum("float64", mshadow::kFloat64)
                .add_enum("float16", mshadow::kFloat16)
                .add_enum("uint8", mshadow::kUint8)
                .add_enum("int32", mshadow::kInt32)
                .describe("Data type");
            DMLC_DECLARE_FIELD(lnz).set_default(0.0f)
                .describe("normalization constant");
        }
    };


    template<typename xpu, typename DType>
    class NceOutputOp: public Operator {
        public:
            explicit NceOutputOp(NceOutputParam& param): param_(param) { }

            virtual void Forward(const OpContext &ctx, 
                    const std::vector<TBlob> &in_data, 
                    const std::vector<OpReqType> &req, 
                    const std::vector<TBlob> &out_data, 
                    const std::vector<TBlob> &aux_args) {
                using namespace mxnet_op;
                if (req[0] == kNullOp) { 
                    return;
                }
                CHECK_NE(req[0], kAddTo);

                mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();

                const TShape& ishape = in_data[NceOutputIdx::kData].shape_;
                const int numlab = ishape[ishape.ndim()-1];
                Shape<2> s2 = Shape2(ishape.Size()/numlab, numlab);

                Tensor<xpu, 2, DType> y = out_data[NceOutputIdx::kOut].get_with_shape<xpu, 2, DType>(s2, s);
                Tensor<xpu, 2, DType> x = in_data[NceOutputIdx::kData].get_with_shape<xpu, 2, DType>(s2, s);
                Tensor<xpu, 2, DType> pn    = in_data[NceOutputIdx::kNegdis].get_with_shape<xpu, 2, DType>(s2, s);

                Kernel<normexp, xpu>::Launch(s, ishape.Size(), y.dptr_, x.dptr_, pn.dptr_, param_.lnz, numlab);
            }


            virtual void Backward(const OpContext &ctx, 
                    const std::vector<TBlob> &out_grad, 
                    const std::vector<TBlob> &in_data, 
                    const std::vector<TBlob> &out_data, 
                    const std::vector<OpReqType> &req, 
                    const std::vector<TBlob> &in_grad, 
                    const std::vector<TBlob> &aux_args) {
                using namespace mshadow;
                using namespace mshadow::expr;

                using namespace mxnet_op;
                if(kNullOp==req[0]) {
                    return;
                }

                mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();

                const TShape& shape= out_data[NceOutputIdx::kOut].shape_;
                const int numlab = shape[shape.ndim()-1];
                Shape<2> s2 = Shape2(shape.Size()/numlab, numlab);

                Tensor<xpu, 2, DType> y     = out_data[NceOutputIdx::kOut].get_with_shape<xpu, 2, DType>(s2, s);
                Tensor<xpu, 2, DType> labwgt= in_data[NceOutputIdx::kWeight].get_with_shape<xpu, 2, DType>(s2, s);
                Tensor<xpu, 2, DType> ograd = in_grad[NceOutputIdx::kData].get_with_shape<xpu, 2, DType>(s2, s);

                // compute data gradient
                Kernel<nceoutGrad, xpu>::Launch(s, shape.Size(), ograd.dptr_, y.dptr_, labwgt.dptr_, numlab);
            }

        private:
            NceOutputParam param_;
    };


#if DMLC_USE_CXX11
    class NceOutputProp: public OperatorProperty {
        public: 
            std::vector<std::string> ListArguments() const override {
                return {"data", "label", "label_weight", "negdis"};
            }

            void Init(const std::vector< std::pair<std::string, std::string> >& kwargs) override {
                param_.Init(kwargs);
            }

            std::map<std::string, std::string> GetParams() const override {
                return param_.__DICT__();
            }

            bool InferShape(std::vector<TShape> *in_shape,
                    std::vector<TShape> *out_shape,
                    std::vector<TShape> *aux_shape) const override {
                using namespace mshadow;
                CHECK_EQ(in_shape->size(), 4U) << "Input: [data, label, label_weight, negdis]";

                const TShape &dshape = in_shape->at(0);
                if (dshape.ndim() ==  0) {
                    return false;
                }
                out_shape->clear();
                out_shape->push_back(dshape);

                return true;
            }

            bool InferType(std::vector<int> *in_type,
                    std::vector<int> *out_type,
                    std::vector<int> *aux_type) const override {
                CHECK_EQ(in_type->size(), 4U);
                int dtype = (*in_type)[0];
                CHECK_GE(dtype, -1) << "First input must have specified type";

                index_t i;
                for(i=1; i<in_type->size(); ++i) {
                    if(-1==(*in_type)[i]) {
                        (*in_type)[i] = dtype;
                    }
                }

                out_type->clear();
                out_type->push_back(dtype);
                return true;
            }

            OperatorProperty* Copy() const override {
                auto ptr = new NceOutputProp();
                ptr->param_ = param_;
                return ptr;
            }

            std::string TypeString() const override {
                return "NceOutput";
            }

            std::vector<int> DeclareBackwardDependency( 
                    const std::vector<int> &out_grad,
                    const std::vector<int> &in_data,
                    const std::vector<int> &out_data) const override {
                return {in_data[NceOutputIdx::kWeight], in_data[NceOutputIdx::kNegdis], out_data[NceOutputIdx::kOut]};
            }

            /*
               std::vector<std::pair<int, void*>> BackwardInplaceOption(
               const std::vector<int> &out_grad,
               const std::vector<int> &in_data,
               const std::vector<int> &out_data,
               const std::vector<int> &in_grad) const override {
               }

               std::vector<std::pair<int, void*>> ForwardInplaceOption(
               const std::vector<int> &in_data,
               const std::vector<int> &out_data) const override {
               }

               std::vector<ResourceRequest> BackwardResource(
               const std::vector<TShape> &in_shape) const override {
               return {ResourceRequest::kTempSpace};
               }
               */

            Operator* CreateOperator(Context ctx) const override {
                LOG(FATAL) << "Not Implemented";
                return NULL;
            }

            Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape, 
                    std::vector<int> *in_type) const override;

        private:
            NceOutputParam param_;
    };
#endif // DMLC_USE_CXX11

} // namespace op
} // namespace mxnet

#endif // MXNET_OPERATOR_NCE_OUTPUT_INL_H_

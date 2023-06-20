#include <nanobind/nanobind.h>

#include <nanobind/stl/vector.h>

#include <ggml.h>

namespace nb = nanobind;

using namespace nb::literals;

// Need to wrap opaque pointer ggml_context * for nanobind
typedef struct ggml_context_p
{
      ggml_context *ctx;
} ggml_context_p;

NB_MODULE(ggml_ext, m)
{
      // #define GGML_FILE_MAGIC   0x67676d6c // "ggml"
      m.attr("GGML_FILE_MAGIC") = GGML_FILE_MAGIC;
      // #define GGML_FILE_VERSION 1
      m.attr("GGML_FILE_VERSION") = GGML_FILE_VERSION;

      // #define GGML_QNT_VERSION        2    // bump this on quantization format changes
      m.attr("GGML_QNT_VERSION") = GGML_QNT_VERSION;
      // #define GGML_QNT_VERSION_FACTOR 1000 // do not change this
      m.attr("GGML_QNT_VERSION_FACTOR") = GGML_QNT_VERSION_FACTOR;

      // #define GGML_MAX_DIMS          4
      m.attr("GGML_MAX_DIMS") = GGML_MAX_DIMS;
      // #define GGML_MAX_NODES         4096
      m.attr("GGML_MAX_NODES") = GGML_MAX_NODES;
      // #define GGML_MAX_PARAMS        256
      m.attr("GGML_MAX_PARAMS") = GGML_MAX_PARAMS;
      // #define GGML_MAX_CONTEXTS      64
      m.attr("GGML_MAX_CONTEXTS") = GGML_MAX_CONTEXTS;
      // #define GGML_MAX_OPT           4
      m.attr("GGML_MAX_OPT") = GGML_MAX_OPT;
      // #define GGML_MAX_NAME          32
      m.attr("GGML_MAX_NAME") = GGML_MAX_NAME;
      // #define GGML_DEFAULT_N_THREADS 4
      m.attr("GGML_DEFAULT_N_THREADS") = GGML_DEFAULT_N_THREADS;

      // #ifdef __ARM_NEON
      //     // we use the built-in 16-bit float type
      //     typedef __fp16 ggml_fp16_t;
      // #else
      //     typedef uint16_t ggml_fp16_t;
      // #endif
      nb::class_<ggml_fp16_t>(m, "ggml_fp16_t");

      // // convert FP16 <-> FP32
      // GGML_API float       ggml_fp16_to_fp32(ggml_fp16_t x);
      m.def("ggml_fp16_to_fp32", &ggml_fp16_to_fp32, nb::arg("x"));
      // GGML_API ggml_fp16_t ggml_fp32_to_fp16(float x);
      m.def("ggml_fp32_to_fp16", &ggml_fp32_to_fp16, nb::arg("x"));

      // GGML_API void ggml_fp16_to_fp32_row(const ggml_fp16_t * x, float * y, size_t n);
      m.def("ggml_fp16_to_fp32_row", &ggml_fp16_to_fp32_row, nb::arg("x"), nb::arg("y"), nb::arg("n"));
      // GGML_API void ggml_fp32_to_fp16_row(const float * x, ggml_fp16_t * y, size_t n);
      m.def("ggml_fp32_to_fp16_row", &ggml_fp32_to_fp16_row, nb::arg("x"), nb::arg("y"), nb::arg("n"));

      // // struct ggml_context;
      nb::class_<ggml_context_p>(m, "ggml_context_p");

      // enum ggml_type {
      //     GGML_TYPE_F32  = 0,
      //     GGML_TYPE_F16  = 1,
      //     GGML_TYPE_Q4_0 = 2,
      //     GGML_TYPE_Q4_1 = 3,
      //     // GGML_TYPE_Q4_2 = 4, support has been removed
      //     // GGML_TYPE_Q4_3 (5) support has been removed
      //     GGML_TYPE_Q5_0 = 6,
      //     GGML_TYPE_Q5_1 = 7,
      //     GGML_TYPE_Q8_0 = 8,
      //     GGML_TYPE_Q8_1 = 9,
      //     GGML_TYPE_I8,
      //     GGML_TYPE_I16,
      //     GGML_TYPE_I32,
      //     GGML_TYPE_COUNT,
      // };
      nb::enum_<ggml_type>(m, "ggml_type")
          .value("GGML_TYPE_F32", GGML_TYPE_F32)
          .value("GGML_TYPE_F16", GGML_TYPE_F16)
          .value("GGML_TYPE_Q4_0", GGML_TYPE_Q4_0)
          .value("GGML_TYPE_Q4_1", GGML_TYPE_Q4_1)
          .value("GGML_TYPE_Q5_0", GGML_TYPE_Q5_0)
          .value("GGML_TYPE_Q5_1", GGML_TYPE_Q5_1)
          .value("GGML_TYPE_Q8_0", GGML_TYPE_Q8_0)
          .value("GGML_TYPE_Q8_1", GGML_TYPE_Q8_1)
          .value("GGML_TYPE_I8", GGML_TYPE_I8)
          .value("GGML_TYPE_I16", GGML_TYPE_I16)
          .value("GGML_TYPE_I32", GGML_TYPE_I32)
          .value("GGML_TYPE_COUNT", GGML_TYPE_COUNT)
          .export_values();

      // enum ggml_backend {
      //     GGML_BACKEND_CPU = 0,
      //     GGML_BACKEND_CUDA = 1,
      //     GGML_BACKEND_CL = 2,
      // };
      nb::enum_<ggml_backend>(m, "ggml_backend")
          .value("GGML_BACKEND_CPU", GGML_BACKEND_CPU)
          .value("GGML_BACKEND_CUDA", GGML_BACKEND_CUDA)
          .value("GGML_BACKEND_CL", GGML_BACKEND_CL)
          .export_values();

      // // model file types
      // enum ggml_ftype {
      //     GGML_FTYPE_UNKNOWN     = -1,
      //     GGML_FTYPE_ALL_F32     = 0,
      //     GGML_FTYPE_MOSTLY_F16  = 1,  // except 1d tensors
      //     GGML_FTYPE_MOSTLY_Q4_0 = 2,  // except 1d tensors
      //     GGML_FTYPE_MOSTLY_Q4_1 = 3,  // except 1d tensors
      //     GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
      //     GGML_FTYPE_MOSTLY_Q8_0 = 7,  // except 1d tensors
      //     GGML_FTYPE_MOSTLY_Q5_0 = 8,  // except 1d tensors
      //     GGML_FTYPE_MOSTLY_Q5_1 = 9,  // except 1d tensors
      // };
      nb::enum_<ggml_ftype>(m, "ggml_ftype")
          .value("GGML_FTYPE_UNKNOWN", GGML_FTYPE_UNKNOWN)
          .value("GGML_FTYPE_ALL_F32", GGML_FTYPE_ALL_F32)
          .value("GGML_FTYPE_MOSTLY_F16", GGML_FTYPE_MOSTLY_F16)
          .value("GGML_FTYPE_MOSTLY_Q4_0", GGML_FTYPE_MOSTLY_Q4_0)
          .value("GGML_FTYPE_MOSTLY_Q4_1", GGML_FTYPE_MOSTLY_Q4_1)
          .value("GGML_FTYPE_MOSTLY_Q4_1_SOME_F16", GGML_FTYPE_MOSTLY_Q4_1_SOME_F16)
          .value("GGML_FTYPE_MOSTLY_Q8_0", GGML_FTYPE_MOSTLY_Q8_0)
          .value("GGML_FTYPE_MOSTLY_Q5_0", GGML_FTYPE_MOSTLY_Q5_0)
          .value("GGML_FTYPE_MOSTLY_Q5_1", GGML_FTYPE_MOSTLY_Q5_1)
          .export_values();

      // // available tensor operations:
      // enum ggml_op {
      //     GGML_OP_NONE = 0,

      //     GGML_OP_DUP,
      //     GGML_OP_ADD,
      //     GGML_OP_ADD1,
      //     GGML_OP_ACC,
      //     GGML_OP_SUB,
      //     GGML_OP_MUL,
      //     GGML_OP_DIV,
      //     GGML_OP_SQR,
      //     GGML_OP_SQRT,
      //     GGML_OP_LOG,
      //     GGML_OP_SUM,
      //     GGML_OP_SUM_ROWS,
      //     GGML_OP_MEAN,
      //     GGML_OP_REPEAT,
      //     GGML_OP_ABS,
      //     GGML_OP_SGN,
      //     GGML_OP_NEG,
      //     GGML_OP_STEP,
      //     GGML_OP_RELU,
      //     GGML_OP_GELU,
      //     GGML_OP_GELU_QUICK,
      //     GGML_OP_SILU,
      //     GGML_OP_SILU_BACK,
      //     GGML_OP_NORM, // normalize
      //     GGML_OP_RMS_NORM,
      //     GGML_OP_RMS_NORM_BACK,

      //     GGML_OP_MUL_MAT,

      //     GGML_OP_SCALE,
      //     GGML_OP_SET,
      //     GGML_OP_CPY,
      //     GGML_OP_CONT,
      //     GGML_OP_RESHAPE,
      //     GGML_OP_VIEW,
      //     GGML_OP_PERMUTE,
      //     GGML_OP_TRANSPOSE,
      //     GGML_OP_GET_ROWS,
      //     GGML_OP_GET_ROWS_BACK,
      //     GGML_OP_DIAG,
      //     GGML_OP_DIAG_MASK_INF,
      //     GGML_OP_DIAG_MASK_ZERO,
      //     GGML_OP_SOFT_MAX,
      //     GGML_OP_ROPE,
      //     GGML_OP_ROPE_BACK,
      //     GGML_OP_ALIBI,
      //     GGML_OP_CLAMP,
      //     GGML_OP_CONV_1D_S1_PH,
      //     GGML_OP_CONV_1D_S2_PH,
      //     GGML_OP_CONV_2D_SK_P0,

      //     GGML_OP_FLASH_ATTN,
      //     GGML_OP_FLASH_FF,
      //     GGML_OP_WIN_PART,
      //     GGML_OP_WIN_UNPART,

      //     GGML_OP_MAP_UNARY,
      //     GGML_OP_MAP_BINARY,

      //     GGML_OP_COUNT,
      // };
      nb::enum_<ggml_op>(m, "ggml_op")
          .value("GGML_OP_NONE", GGML_OP_NONE)

          .value("GGML_OP_DUP", GGML_OP_DUP)
          .value("GGML_OP_ADD", GGML_OP_ADD)
          .value("GGML_OP_ADD1", GGML_OP_ADD1)
          .value("GGML_OP_ACC", GGML_OP_ACC)
          .value("GGML_OP_SUB", GGML_OP_SUB)
          .value("GGML_OP_MUL", GGML_OP_MUL)
          .value("GGML_OP_DIV", GGML_OP_DIV)
          .value("GGML_OP_SQR", GGML_OP_SQR)
          .value("GGML_OP_SQRT", GGML_OP_SQRT)
          .value("GGML_OP_LOG", GGML_OP_LOG)
          .value("GGML_OP_SUM", GGML_OP_SUM)
          .value("GGML_OP_SUM_ROWS", GGML_OP_SUM_ROWS)
          .value("GGML_OP_MEAN", GGML_OP_MEAN)
          .value("GGML_OP_REPEAT", GGML_OP_REPEAT)
          .value("GGML_OP_ABS", GGML_OP_ABS)
          .value("GGML_OP_SGN", GGML_OP_SGN)
          .value("GGML_OP_NEG", GGML_OP_NEG)
          .value("GGML_OP_STEP", GGML_OP_STEP)
          .value("GGML_OP_RELU", GGML_OP_RELU)
          .value("GGML_OP_GELU", GGML_OP_GELU)
          .value("GGML_OP_GELU_QUICK", GGML_OP_GELU_QUICK)
          .value("GGML_OP_SILU", GGML_OP_SILU)
          .value("GGML_OP_SILU_BACK", GGML_OP_SILU_BACK)
          .value("GGML_OP_NORM", GGML_OP_NORM)
          .value("GGML_OP_RMS_NORM", GGML_OP_RMS_NORM)
          .value("GGML_OP_RMS_NORM_BACK", GGML_OP_RMS_NORM_BACK)

          .value("GGML_OP_MUL_MAT", GGML_OP_MUL_MAT)

          .value("GGML_OP_SCALE", GGML_OP_SCALE)
          .value("GGML_OP_SET", GGML_OP_SET)
          .value("GGML_OP_CPY", GGML_OP_CPY)
          .value("GGML_OP_CONT", GGML_OP_CONT)
          .value("GGML_OP_RESHAPE", GGML_OP_RESHAPE)
          .value("GGML_OP_VIEW", GGML_OP_VIEW)
          .value("GGML_OP_PERMUTE", GGML_OP_PERMUTE)
          .value("GGML_OP_TRANSPOSE", GGML_OP_TRANSPOSE)
          .value("GGML_OP_GET_ROWS", GGML_OP_GET_ROWS)
          .value("GGML_OP_GET_ROWS_BACK", GGML_OP_GET_ROWS_BACK)
          .value("GGML_OP_DIAG", GGML_OP_DIAG)
          .value("GGML_OP_DIAG_MASK_INF", GGML_OP_DIAG_MASK_INF)
          .value("GGML_OP_DIAG_MASK_ZERO", GGML_OP_DIAG_MASK_ZERO)
          .value("GGML_OP_SOFT_MAX", GGML_OP_SOFT_MAX)
          .value("GGML_OP_ROPE", GGML_OP_ROPE)
          .value("GGML_OP_ROPE_BACK", GGML_OP_ROPE_BACK)
          .value("GGML_OP_ALIBI", GGML_OP_ALIBI)
          .value("GGML_OP_CLAMP", GGML_OP_CLAMP)
          .value("GGML_OP_CONV_1D_S1_PH", GGML_OP_CONV_1D_S1_PH)
          .value("GGML_OP_CONV_1D_S2_PH", GGML_OP_CONV_1D_S2_PH)
          .value("GGML_OP_CONV_2D_SK_P0", GGML_OP_CONV_2D_SK_P0)

          .value("GGML_OP_FLASH_ATTN", GGML_OP_FLASH_ATTN)
          .value("GGML_OP_FLASH_FF", GGML_OP_FLASH_FF)
          .value("GGML_OP_WIN_PART", GGML_OP_WIN_PART)
          .value("GGML_OP_WIN_UNPART", GGML_OP_WIN_UNPART)

          .value("GGML_OP_MAP_UNARY", GGML_OP_MAP_UNARY)
          .value("GGML_OP_MAP_BINARY", GGML_OP_MAP_BINARY)

          .value("GGML_OP_COUNT", GGML_OP_COUNT)
          .export_values();

      // // ggml object
      // struct ggml_object {
      //     size_t offs;
      //     size_t size;

      //     struct ggml_object * next;

      //     char padding[8];
      // };
      nb::class_<ggml_object>(m, "ggml_object");

      // static const size_t GGML_OBJECT_SIZE = sizeof(struct ggml_object);
      m.attr("GGML_OBJECT_SIZE") = GGML_OBJECT_SIZE;

      // // n-dimensional tensor
      // struct ggml_tensor {
      //     enum ggml_type    type;
      //     enum ggml_backend backend;

      //     int     n_dims;
      //     int64_t ne[GGML_MAX_DIMS]; // number of elements
      //     size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
      //                                // nb[0] = sizeof(type)
      //                                // nb[1] = nb[0]   * ne[0] + padding
      //                                // nb[i] = nb[i-1] * ne[i-1]

      //     // compute data
      //     enum ggml_op op;

      //     bool is_param;

      //     struct ggml_tensor * grad;
      //     struct ggml_tensor * src0;
      //     struct ggml_tensor * src1;
      //     struct ggml_tensor * opt[GGML_MAX_OPT];

      //     // thread scheduling
      //     int n_tasks;

      //     // performance
      //     int     perf_runs;
      //     int64_t perf_cycles;
      //     int64_t perf_time_us;

      //     void * data;

      //     char name[GGML_MAX_NAME];

      //     char padding[16];
      // };
      nb::class_<ggml_tensor>(m, "ggml_tensor")
          .def_rw("type", &ggml_tensor::type)
          .def_rw("backend", &ggml_tensor::backend)

          .def_rw("n_dims", &ggml_tensor::n_dims)

          .def_prop_rw(
              "ne", [](ggml_tensor &self) -> std::vector<int64_t>
              {
                    std::vector<int64_t> ne(self.n_dims);
                    for (int i = 0; i < self.n_dims; i++)
                          ne[i] = self.ne[i];
                    return ne; },
              [](ggml_tensor &self, std::vector<int64_t> ne)
              {
                    for (int i = 0; i < self.n_dims; i++)
                          self.ne[i] = ne[i];
              })
          .def_prop_rw(
              "nb", [](ggml_tensor &self) -> std::vector<size_t>
              {
                    std::vector<size_t> nb(self.n_dims);
                    for (int i = 0; i < self.n_dims; i++)
                          nb[i] = self.nb[i];
                    return nb; },
              [](ggml_tensor &self, std::vector<size_t> nb)
              {
                    for (int i = 0; i < self.n_dims; i++)
                          self.nb[i] = nb[i];
              })
          .def_rw("op", &ggml_tensor::op)
          .def_rw("is_param", &ggml_tensor::is_param)
          .def_rw("grad", &ggml_tensor::grad)
          .def_rw("src0", &ggml_tensor::src0)
          .def_rw("src1", &ggml_tensor::src1)
          .def_prop_rw(
              "opt", [](ggml_tensor &self) -> std::vector<ggml_tensor *>
              {
                    std::vector<ggml_tensor *> opt(GGML_MAX_OPT);
                    for (int i = 0; i < GGML_MAX_OPT; i++)
                          opt[i] = self.opt[i];
                    return opt; },
              [](ggml_tensor &self, std::vector<ggml_tensor *> opt)
              {
                    for (int i = 0; i < GGML_MAX_OPT; i++)
                          self.opt[i] = opt[i];
              })
          .def_rw("n_tasks", &ggml_tensor::n_tasks)
          .def_rw("perf_runs", &ggml_tensor::perf_runs)
          .def_rw("perf_cycles", &ggml_tensor::perf_cycles)
          .def_rw("perf_time_us", &ggml_tensor::perf_time_us)
          .def_rw("data", &ggml_tensor::data)
          .def_prop_rw(
              "name", [](ggml_tensor &self) -> std::string
              { return std::string(self.name); },
              [](ggml_tensor &self, std::string name)
              {
                    strcpy(self.name, name.c_str());
              })
          .def_prop_rw(
              "padding", [](ggml_tensor &self) -> std::vector<char>
              {
                    std::vector<char> padding(16);
                    for (int i = 0; i < 16; i++)
                          padding[i] = self.padding[i];
                    return padding; },
              [](ggml_tensor &self, std::vector<char> padding)
              {
                    for (int i = 0; i < 16; i++)
                          self.padding[i] = padding[i];
              });

      // static const size_t GGML_TENSOR_SIZE = sizeof(struct ggml_tensor);
      m.attr("GGML_TENSOR_SIZE") = GGML_TENSOR_SIZE;

      // // computation graph
      // struct ggml_cgraph {
      //     int n_nodes;
      //     int n_leafs;
      //     int n_threads;

      //     size_t work_size;
      //     struct ggml_tensor * work;

      //     struct ggml_tensor * nodes[GGML_MAX_NODES];
      //     struct ggml_tensor * grads[GGML_MAX_NODES];
      //     struct ggml_tensor * leafs[GGML_MAX_NODES];

      //     // performance
      //     int     perf_runs;
      //     int64_t perf_cycles;
      //     int64_t perf_time_us;
      // };
      nb::class_<ggml_cgraph>(m, "ggml_cgraph");

      // // scratch buffer
      // struct ggml_scratch {
      //     size_t offs;
      //     size_t size;
      //     void * data;
      // };
      nb::class_<ggml_scratch>(m, "ggml_scratch");

      // struct ggml_init_params {
      //     // memory pool
      //     size_t mem_size;   // bytes
      //     void * mem_buffer; // if NULL, memory will be allocated internally
      //     bool   no_alloc;   // don't allocate memory for the tensor data
      // };
      nb::class_<ggml_init_params>(m, "ggml_init_params")
          .def(
              "__init__", [](ggml_init_params &self, size_t mem_size, void *mem_buffer, bool no_alloc)
              {
            self.mem_size = mem_size;
            self.mem_buffer = mem_buffer;
            self.no_alloc = no_alloc; },
              nb::arg("mem_size") = 0, nb::arg("mem_buffer") = nullptr, nb::arg("no_alloc") = true)
          .def_rw("mem_size", &ggml_init_params::mem_size)
          .def_rw("mem_buffer", &ggml_init_params::mem_buffer)
          .def_rw("no_alloc", &ggml_init_params::no_alloc);

      // // misc

      // GGML_API void    ggml_time_init(void); // call this once at the beginning of the program
      m.def("ggml_time_init", &ggml_time_init);
      // GGML_API int64_t ggml_time_ms(void);
      m.def("ggml_time_ms", &ggml_time_ms);
      // GGML_API int64_t ggml_time_us(void);
      m.def("ggml_time_us", &ggml_time_us);
      // GGML_API int64_t ggml_cycles(void);
      m.def("ggml_cycles", &ggml_cycles);
      // GGML_API int64_t ggml_cycles_per_ms(void);
      m.def("ggml_cycles_per_ms", &ggml_cycles_per_ms);

      // GGML_API void    ggml_print_object (const struct ggml_object * obj);
      m.def("ggml_print_object", &ggml_print_object);
      // GGML_API void    ggml_print_objects(const struct ggml_context * ctx);
      m.def("ggml_print_objects", [](ggml_context_p ctx_p)
            { ggml_print_objects(ctx_p.ctx); });

      // GGML_API int64_t ggml_nelements(const struct ggml_tensor * tensor);
      m.def("ggml_nelements", &ggml_nelements);
      // GGML_API size_t  ggml_nbytes   (const struct ggml_tensor * tensor);
      m.def("ggml_nbytes", &ggml_nbytes);

      // GGML_API int     ggml_blck_size (enum ggml_type type);
      m.def("ggml_blck_size", &ggml_blck_size);
      // GGML_API size_t  ggml_type_size (enum ggml_type type); // size in bytes for all elements in a block
      m.def("ggml_type_size", &ggml_type_size);
      // GGML_API float   ggml_type_sizef(enum ggml_type type); // ggml_type_size()/ggml_blck_size() as float
      m.def("ggml_type_sizef", &ggml_type_sizef);

      // GGML_API const char * ggml_type_name(enum ggml_type type);
      m.def("ggml_type_name", &ggml_type_name);
      // GGML_API const char * ggml_op_name  (enum ggml_op   op);
      m.def("ggml_op_name", &ggml_op_name);

      // GGML_API size_t  ggml_element_size(const struct ggml_tensor * tensor);
      m.def("ggml_element_size", &ggml_element_size);

      // GGML_API bool    ggml_is_quantized(enum ggml_type type);
      m.def("ggml_is_quantized", &ggml_is_quantized);

      // // TODO: temporary until model loading of ggml examples is refactored
      // GGML_API enum ggml_type ggml_ftype_to_ggml_type(enum ggml_ftype ftype);
      m.def("ggml_ftype_to_ggml_type", &ggml_ftype_to_ggml_type);

      // // use this to compute the memory overhead of a tensor
      // GGML_API size_t ggml_tensor_overhead(void);
      m.def("ggml_tensor_overhead", &ggml_tensor_overhead);

      // // main

      // GGML_API struct ggml_context * ggml_init(struct ggml_init_params params);
      // m.def("ggml_init", &ggml_init, nb::arg("params"));
      m.def(
          "ggml_init", [](ggml_init_params params)
          { return ggml_context_p{ggml_init(params)}; },
          nb::arg("params"));

      // GGML_API void    ggml_free(struct ggml_context * ctx);
      m.def(
          "ggml_free", [](ggml_context_p ctx_p)
          { return ggml_free(ctx_p.ctx); },
          nb::arg("ctx"));

      // GGML_API size_t  ggml_used_mem(const struct ggml_context * ctx);
      m.def(
          "ggml_used_mem", [](ggml_context_p ctx_p)
          { return ggml_used_mem(ctx_p.ctx); },
          nb::arg("ctx"));

      // GGML_API size_t  ggml_set_scratch (struct ggml_context * ctx, struct ggml_scratch scratch);
      m.def(
          "ggml_set_scratch", [](ggml_context_p ctx_p, ggml_scratch scratch)
          { return ggml_set_scratch(ctx_p.ctx, scratch); },
          nb::arg("ctx"), nb::arg("scratch"));
      // GGML_API void    ggml_set_no_alloc(struct ggml_context * ctx, bool no_alloc);
      m.def(
          "ggml_set_no_alloc", [](ggml_context_p ctx_p, bool no_alloc)
          { ggml_set_no_alloc(ctx_p.ctx, no_alloc); },
          nb::arg("ctx"), nb::arg("no_alloc"));

      // GGML_API void *  ggml_get_mem_buffer(struct ggml_context * ctx);
      m.def("ggml_get_mem_buffer", [](ggml_context_p ctx_p)
            { return ggml_get_mem_buffer(ctx_p.ctx); });
      // GGML_API size_t  ggml_get_mem_size  (struct ggml_context * ctx);
      m.def("ggml_get_mem_size", [](ggml_context_p ctx_p)
            { return ggml_get_mem_size(ctx_p.ctx); });

      // GGML_API struct ggml_tensor * ggml_new_tensor(
      //         struct ggml_context * ctx,
      //         enum   ggml_type type,
      //         int    n_dims,
      //         const int64_t *ne);
      m.def(
          "ggml_new_tensor", [](ggml_context_p ctx_p, ggml_type type, int n_dims, const int64_t *ne)
          { return ggml_new_tensor(ctx_p.ctx, type, n_dims, ne); },
          nb::arg("ctx"), nb::arg("type"), nb::arg("n_dims"), nb::arg("ne"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_new_tensor_1d(
      //         struct ggml_context * ctx,
      //         enum   ggml_type type,
      //         int64_t ne0);
      m.def(
          "ggml_new_tensor_1d", [](ggml_context_p ctx_p, ggml_type type, int64_t ne0)
          { return ggml_new_tensor_1d(ctx_p.ctx, type, ne0); },
          nb::arg("ctx"), nb::arg("type"), nb::arg("ne0"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_new_tensor_2d(
      //         struct ggml_context * ctx,
      //         enum   ggml_type type,
      //         int64_t ne0,
      //         int64_t ne1);
      m.def(
          "ggml_new_tensor_2d", [](ggml_context_p ctx_p, ggml_type type, int64_t ne0, int64_t ne1)
          { return ggml_new_tensor_2d(ctx_p.ctx, type, ne0, ne1); },
          nb::arg("ctx"), nb::arg("type"), nb::arg("ne0"), nb::arg("ne1"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_new_tensor_3d(
      //         struct ggml_context * ctx,
      //         enum   ggml_type type,
      //         int64_t ne0,
      //         int64_t ne1,
      //         int64_t ne2);
      m.def(
          "ggml_new_tensor_3d", [](ggml_context_p ctx_p, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2)
          { return ggml_new_tensor_3d(ctx_p.ctx, type, ne0, ne1, ne2); },
          nb::arg("ctx"), nb::arg("type"), nb::arg("ne0"), nb::arg("ne1"), nb::arg("ne2"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_new_tensor_4d(
      //         struct ggml_context * ctx,
      //         enum   ggml_type type,
      //         int64_t ne0,
      //         int64_t ne1,
      //         int64_t ne2,
      //         int64_t ne3);
      m.def(
          "ggml_new_tensor_4d", [](ggml_context_p ctx_p, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3)
          { return ggml_new_tensor_4d(ctx_p.ctx, type, ne0, ne1, ne2, ne3); },
          nb::arg("ctx"), nb::arg("type"), nb::arg("ne0"), nb::arg("ne1"), nb::arg("ne2"), nb::arg("ne3"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value);
      m.def(
          "ggml_new_i32", [](ggml_context_p ctx_p, int32_t value)
          { return ggml_new_i32(ctx_p.ctx, value); },
          nb::arg("ctx"), nb::arg("value"), nb::rv_policy::reference);
      // GGML_API struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value);
      m.def(
          "ggml_new_f32", [](ggml_context_p ctx_p, float value)
          { return ggml_new_f32(ctx_p.ctx, value); },
          nb::arg("ctx"), nb::arg("value"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_dup_tensor (struct ggml_context * ctx, const struct ggml_tensor * src);
      m.def(
          "ggml_dup_tensor", [](ggml_context_p ctx_p, const ggml_tensor *src)
          { return ggml_dup_tensor(ctx_p.ctx, src); },
          nb::arg("ctx"), nb::arg("src"), nb::rv_policy::reference);
      // GGML_API struct ggml_tensor * ggml_view_tensor(struct ggml_context * ctx, const struct ggml_tensor * src);
      m.def(
          "ggml_view_tensor", [](ggml_context_p ctx_p, const ggml_tensor *src)
          { return ggml_view_tensor(ctx_p.ctx, src); },
          nb::arg("ctx"), nb::arg("src"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_get_tensor(struct ggml_context * ctx, const char * name);
      m.def(
          "ggml_get_tensor", [](ggml_context_p ctx_p, const char *name)
          { return ggml_get_tensor(ctx_p.ctx, name); },
          nb::arg("ctx"), nb::arg("name"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_set_zero(struct ggml_tensor * tensor);
      m.def("ggml_set_zero", &ggml_set_zero, nb::arg("tensor"), nb::rv_policy::reference);
      // GGML_API struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value);
      m.def("ggml_set_i32", &ggml_set_i32, nb::arg("tensor"), nb::arg("value"), nb::rv_policy::reference);
      // GGML_API struct ggml_tensor * ggml_set_f32 (struct ggml_tensor * tensor, float value);
      m.def("ggml_set_f32", &ggml_set_f32, nb::arg("tensor"), nb::arg("value"), nb::rv_policy::reference);

      // GGML_API int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);
      m.def("ggml_get_i32_1d", &ggml_get_i32_1d, nb::arg("tensor"), nb::arg("i"));
      // GGML_API void    ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);
      m.def("ggml_set_i32_1d", &ggml_set_i32_1d, nb::arg("tensor"), nb::arg("i"), nb::arg("value"));

      // GGML_API float   ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);
      m.def("ggml_get_f32_1d", &ggml_get_f32_1d, nb::arg("tensor"), nb::arg("i"));
      // GGML_API void    ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);
      m.def("ggml_set_f32_1d", &ggml_set_f32_1d, nb::arg("tensor"), nb::arg("i"), nb::arg("value"));

      // GGML_API void *  ggml_get_data    (const struct ggml_tensor * tensor);
      m.def("ggml_get_data", &ggml_get_data, nb::arg("tensor"), nb::rv_policy::reference);
      // GGML_API float * ggml_get_data_f32(const struct ggml_tensor * tensor);
      m.def("ggml_get_data_f32", &ggml_get_data_f32, nb::arg("tensor"), nb::rv_policy::reference);

      // GGML_API const char *         ggml_get_name(const struct ggml_tensor * tensor);
      m.def("ggml_get_name", &ggml_get_name, nb::arg("tensor"), nb::rv_policy::reference);
      // GGML_API struct ggml_tensor * ggml_set_name(struct ggml_tensor * tensor, const char * name);
      m.def("ggml_set_name", &ggml_set_name, nb::arg("tensor"), nb::arg("name"), nb::rv_policy::reference);

      // //
      // // operations on tensors with backpropagation
      // //

      // GGML_API struct ggml_tensor * ggml_dup(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_dup", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_dup(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_add(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         struct ggml_tensor  * b);
      m.def(
          "ggml_add", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_tensor *b)
          { return ggml_add(ctx_p.ctx, a, b); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("b"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_add_inplace(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         struct ggml_tensor  * b);
      m.def(
          "ggml_add_inplace", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_tensor *b)
          { return ggml_add_inplace(ctx_p.ctx, a, b); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("b"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_add1(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         struct ggml_tensor  * b);
      m.def(
          "ggml_add1", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_tensor *b)
          { return ggml_add1(ctx_p.ctx, a, b); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("b"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_add1_inplace(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         struct ggml_tensor  * b);
      m.def(
          "ggml_add1_inplace", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_tensor *b)
          { return ggml_add1_inplace(ctx_p.ctx, a, b); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("b"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_acc(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         struct ggml_tensor  * b,
      //         size_t                nb1,
      //         size_t                nb2,
      //         size_t                nb3,
      //         size_t                offset);
      m.def(
          "ggml_acc", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_tensor *b, size_t nb1, size_t nb2, size_t nb3, size_t offset)
          { return ggml_acc(ctx_p.ctx, a, b, nb1, nb2, nb3, offset); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("b"), nb::arg("nb1"), nb::arg("nb2"), nb::arg("nb3"), nb::arg("offset"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_acc_inplace(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         struct ggml_tensor  * b,
      //         size_t                nb1,
      //         size_t                nb2,
      //         size_t                nb3,
      //         size_t                offset);
      m.def(
          "ggml_acc_inplace", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_tensor *b, size_t nb1, size_t nb2, size_t nb3, size_t offset)
          { return ggml_acc_inplace(ctx_p.ctx, a, b, nb1, nb2, nb3, offset); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("b"), nb::arg("nb1"), nb::arg("nb2"), nb::arg("nb3"), nb::arg("offset"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_sub(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         struct ggml_tensor  * b);
      m.def(
          "ggml_sub", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_tensor *b)
          { return ggml_sub(ctx_p.ctx, a, b); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("b"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_sub_inplace(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         struct ggml_tensor  * b);
      m.def(
          "ggml_sub_inplace", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_tensor *b)
          { return ggml_sub_inplace(ctx_p.ctx, a, b); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("b"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_mul(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         struct ggml_tensor  * b);
      m.def(
          "ggml_mul", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_tensor *b)
          { return ggml_mul(ctx_p.ctx, a, b); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("b"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_mul_inplace(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         struct ggml_tensor  * b);
      m.def(
          "ggml_mul_inplace", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_tensor *b)
          { return ggml_mul_inplace(ctx_p.ctx, a, b); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("b"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_div(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         struct ggml_tensor  * b);
      m.def(
          "ggml_div", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_tensor *b)
          { return ggml_div(ctx_p.ctx, a, b); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("b"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_div_inplace(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         struct ggml_tensor  * b);
      m.def(
          "ggml_div_inplace", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_tensor *b)
          { return ggml_div_inplace(ctx_p.ctx, a, b); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("b"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_sqr(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_sqr", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_sqr(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_sqr_inplace(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_sqr_inplace", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_sqr_inplace(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_sqrt(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_sqrt", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_sqrt(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_sqrt_inplace(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_sqrt_inplace", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_sqrt_inplace(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_log(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_log", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_log(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_log_inplace(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_log_inplace", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_log_inplace(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // // return scalar
      // GGML_API struct ggml_tensor * ggml_sum(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_sum", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_sum(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // // sums along rows, with input shape [a,b,c,d] return shape [1,b,c,d]
      // GGML_API struct ggml_tensor * ggml_sum_rows(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_sum_rows", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_sum_rows(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // // mean along rows
      // GGML_API struct ggml_tensor * ggml_mean(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_mean", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_mean(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // // if a is the same shape as b, and a is not parameter, return a
      // // otherwise, return a new tensor: repeat(a) to fit in b
      // GGML_API struct ggml_tensor * ggml_repeat(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         struct ggml_tensor  * b);
      m.def(
          "ggml_repeat", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_tensor *b)
          { return ggml_repeat(ctx_p.ctx, a, b); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("b"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_abs(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_abs", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_abs(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_abs_inplace(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_abs_inplace", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_abs_inplace(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_sgn(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_sgn", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_sgn(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_sgn_inplace(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_sgn_inplace", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_sgn_inplace(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_neg(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_neg", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_neg(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_neg_inplace(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_neg_inplace", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_neg_inplace(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_step(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_step", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_step(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_step_inplace(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_step_inplace", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_step_inplace(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_relu(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_relu", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_relu(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_relu_inplace(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_relu_inplace", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_relu_inplace(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // // TODO: double-check this computation is correct
      // GGML_API struct ggml_tensor * ggml_gelu(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_gelu", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_gelu(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_gelu_inplace(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_gelu_inplace", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_gelu_inplace(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_gelu_quick(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_gelu_quick", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_gelu_quick(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_gelu_quick_inplace(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_gelu_quick_inplace", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_gelu_quick_inplace(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_silu(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_silu", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_silu(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_silu_inplace(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_silu_inplace", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_silu_inplace(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // // a - x
      // // b - dy
      // GGML_API struct ggml_tensor * ggml_silu_back(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         struct ggml_tensor  * b);
      m.def(
          "ggml_silu_back", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_tensor *b)
          { return ggml_silu_back(ctx_p.ctx, a, b); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("b"), nb::rv_policy::reference);

      // // normalize along rows
      // // TODO: eps is hardcoded to 1e-5 for now
      // GGML_API struct ggml_tensor * ggml_norm(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_norm", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_norm(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_norm_inplace(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_norm_inplace", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_norm_inplace(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_rms_norm(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_rms_norm", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_rms_norm(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_rms_norm_inplace(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_rms_norm_inplace", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_rms_norm_inplace(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // // a - x
      // // b - dy
      // GGML_API struct ggml_tensor * ggml_rms_norm_back(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         struct ggml_tensor  * b);
      m.def(
          "ggml_rms_norm_back", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_tensor *b)
          { return ggml_rms_norm_back(ctx_p.ctx, a, b); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("b"), nb::rv_policy::reference);

      // // A: m rows, n columns
      // // B: p rows, n columns (i.e. we transpose it internally)
      // // result is m columns, p rows
      // GGML_API struct ggml_tensor * ggml_mul_mat(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         struct ggml_tensor  * b);
      m.def(
          "ggml_mul_mat", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_tensor *b)
          { return ggml_mul_mat(ctx_p.ctx, a, b); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("b"), nb::rv_policy::reference);

      // //
      // // operations on tensors without backpropagation
      // //

      // GGML_API struct ggml_tensor * ggml_scale(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         struct ggml_tensor  * b);
      m.def(
          "ggml_scale", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_tensor *b)
          { return ggml_scale(ctx_p.ctx, a, b); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("b"), nb::rv_policy::reference);

      // // in-place, returns view(a)
      // GGML_API struct ggml_tensor * ggml_scale_inplace(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         struct ggml_tensor  * b);
      m.def(
          "ggml_scale_inplace", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_tensor *b)
          { return ggml_scale_inplace(ctx_p.ctx, a, b); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("b"), nb::rv_policy::reference);

      // // b -> view(a,offset,nb1,nb2,3), return modified a
      // GGML_API struct ggml_tensor * ggml_set(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         struct ggml_tensor  * b,
      //         size_t                nb1,
      //         size_t                nb2,
      //         size_t                nb3,
      //         size_t                offset);
      m.def(
          "ggml_set", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_tensor *b, size_t nb1, size_t nb2, size_t nb3, size_t offset)
          { return ggml_set(ctx_p.ctx, a, b, nb1, nb2, nb3, offset); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("b"), nb::arg("nb1"), nb::arg("nb2"), nb::arg("nb3"), nb::arg("offset"), nb::rv_policy::reference);

      // // b -> view(a,offset,nb1,nb2,3), return view(a)
      // GGML_API struct ggml_tensor * ggml_set_inplace(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         struct ggml_tensor  * b,
      //         size_t                nb1,
      //         size_t                nb2,
      //         size_t                nb3,
      //         size_t                offset);
      m.def(
          "ggml_set_inplace", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_tensor *b, size_t nb1, size_t nb2, size_t nb3, size_t offset)
          { return ggml_set_inplace(ctx_p.ctx, a, b, nb1, nb2, nb3, offset); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("b"), nb::arg("nb1"), nb::arg("nb2"), nb::arg("nb3"), nb::arg("offset"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_set_1d(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         struct ggml_tensor  * b,
      //         size_t                offset);
      m.def(
          "ggml_set_1d", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_tensor *b, size_t offset)
          { return ggml_set_1d(ctx_p.ctx, a, b, offset); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("b"), nb::arg("offset"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_set_1d_inplace(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         struct ggml_tensor  * b,
      //         size_t                offset);
      m.def(
          "ggml_set_1d_inplace", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_tensor *b, size_t offset)
          { return ggml_set_1d_inplace(ctx_p.ctx, a, b, offset); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("b"), nb::arg("offset"), nb::rv_policy::reference);

      // // b -> view(a,offset,nb1,nb2,3), return modified a
      // GGML_API struct ggml_tensor * ggml_set_2d(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         struct ggml_tensor  * b,
      //         size_t                nb1,
      //         size_t                offset);
      m.def(
          "ggml_set_2d", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_tensor *b, size_t nb1, size_t offset)
          { return ggml_set_2d(ctx_p.ctx, a, b, nb1, offset); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("b"), nb::arg("nb1"), nb::arg("offset"), nb::rv_policy::reference);

      // // b -> view(a,offset,nb1,nb2,3), return view(a)
      // GGML_API struct ggml_tensor * ggml_set_2d_inplace(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         struct ggml_tensor  * b,
      //         size_t                nb1,
      //         size_t                offset);
      m.def(
          "ggml_set_2d_inplace", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_tensor *b, size_t nb1, size_t offset)
          { return ggml_set_2d_inplace(ctx_p.ctx, a, b, nb1, offset); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("b"), nb::arg("nb1"), nb::arg("offset"), nb::rv_policy::reference);

      // // a -> b, return view(b)
      // GGML_API struct ggml_tensor * ggml_cpy(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         struct ggml_tensor  * b);
      m.def(
          "ggml_cpy", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_tensor *b)
          { return ggml_cpy(ctx_p.ctx, a, b); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("b"), nb::rv_policy::reference);

      // // make contiguous
      // GGML_API struct ggml_tensor * ggml_cont(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_cont", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_cont(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // // return view(a), b specifies the new shape
      // // TODO: when we start computing gradient, make a copy instead of view
      // GGML_API struct ggml_tensor * ggml_reshape(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         struct ggml_tensor  * b);
      m.def(
          "ggml_reshape", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_tensor *b)
          { return ggml_reshape(ctx_p.ctx, a, b); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("b"), nb::rv_policy::reference);

      // // return view(a)
      // // TODO: when we start computing gradient, make a copy instead of view
      // GGML_API struct ggml_tensor * ggml_reshape_1d(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         int64_t               ne0);
      m.def(
          "ggml_reshape_1d", [](ggml_context_p ctx_p, ggml_tensor *a, int64_t ne0)
          { return ggml_reshape_1d(ctx_p.ctx, a, ne0); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("ne0"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_reshape_2d(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         int64_t               ne0,
      //         int64_t               ne1);
      m.def(
          "ggml_reshape_2d", [](ggml_context_p ctx_p, ggml_tensor *a, int64_t ne0, int64_t ne1)
          { return ggml_reshape_2d(ctx_p.ctx, a, ne0, ne1); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("ne0"), nb::arg("ne1"), nb::rv_policy::reference);

      // // return view(a)
      // // TODO: when we start computing gradient, make a copy instead of view
      // GGML_API struct ggml_tensor * ggml_reshape_3d(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         int64_t               ne0,
      //         int64_t               ne1,
      //         int64_t               ne2);
      m.def(
          "ggml_reshape_3d", [](ggml_context_p ctx_p, ggml_tensor *a, int64_t ne0, int64_t ne1, int64_t ne2)
          { return ggml_reshape_3d(ctx_p.ctx, a, ne0, ne1, ne2); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("ne0"), nb::arg("ne1"), nb::arg("ne2"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_reshape_4d(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         int64_t               ne0,
      //         int64_t               ne1,
      //         int64_t               ne2,
      //         int64_t               ne3);
      m.def(
          "ggml_reshape_4d", [](ggml_context_p ctx_p, ggml_tensor *a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3)
          { return ggml_reshape_4d(ctx_p.ctx, a, ne0, ne1, ne2, ne3); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("ne0"), nb::arg("ne1"), nb::arg("ne2"), nb::arg("ne3"), nb::rv_policy::reference);

      // // offset in bytes
      // GGML_API struct ggml_tensor * ggml_view_1d(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         int64_t               ne0,
      //         size_t                offset);
      m.def(
          "ggml_view_1d", [](ggml_context_p ctx_p, ggml_tensor *a, int64_t ne0, size_t offset)
          { return ggml_view_1d(ctx_p.ctx, a, ne0, offset); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("ne0"), nb::arg("offset"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_view_2d(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         int64_t               ne0,
      //         int64_t               ne1,
      //         size_t                nb1, // row stride in bytes
      //         size_t                offset);
      m.def(
          "ggml_view_2d", [](ggml_context_p ctx_p, ggml_tensor *a, int64_t ne0, int64_t ne1, size_t nb1, size_t offset)
          { return ggml_view_2d(ctx_p.ctx, a, ne0, ne1, nb1, offset); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("ne0"), nb::arg("ne1"), nb::arg("nb1"), nb::arg("offset"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_view_3d(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         int64_t               ne0,
      //         int64_t               ne1,
      //         int64_t               ne2,
      //         size_t                nb1, // row   stride in bytes
      //         size_t                nb2, // slice stride in bytes
      //         size_t                offset);
      m.def(
          "ggml_view_3d", [](ggml_context_p ctx_p, ggml_tensor *a, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb1, size_t nb2, size_t offset)
          { return ggml_view_3d(ctx_p.ctx, a, ne0, ne1, ne2, nb1, nb2, offset); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("ne0"), nb::arg("ne1"), nb::arg("ne2"), nb::arg("nb1"), nb::arg("nb2"), nb::arg("offset"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_view_4d(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         int64_t               ne0,
      //         int64_t               ne1,
      //         int64_t               ne2,
      //         int64_t               ne3,
      //         size_t                nb1, // row   stride in bytes
      //         size_t                nb2, // slice stride in bytes
      //         size_t                nb3,
      //         size_t                offset);
      m.def(
          "ggml_view_4d", [](ggml_context_p ctx_p, ggml_tensor *a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t nb1, size_t nb2, size_t nb3, size_t offset)
          { return ggml_view_4d(ctx_p.ctx, a, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("ne0"), nb::arg("ne1"), nb::arg("ne2"), nb::arg("ne3"), nb::arg("nb1"), nb::arg("nb2"), nb::arg("nb3"), nb::arg("offset"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_permute(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         int                   axis0,
      //         int                   axis1,
      //         int                   axis2,
      //         int                   axis3);
      m.def(
          "ggml_permute", [](ggml_context_p ctx_p, ggml_tensor *a, int axis0, int axis1, int axis2, int axis3)
          { return ggml_permute(ctx_p.ctx, a, axis0, axis1, axis2, axis3); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("axis0"), nb::arg("axis1"), nb::arg("axis2"), nb::arg("axis3"), nb::rv_policy::reference);

      // // alias for ggml_permute(ctx, a, 1, 0, 2, 3)
      // GGML_API struct ggml_tensor * ggml_transpose(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_transpose", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_transpose(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_get_rows(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         struct ggml_tensor  * b);
      m.def(
          "ggml_get_rows", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_tensor *b)
          { return ggml_get_rows(ctx_p.ctx, a, b); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("b"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_get_rows_back(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         struct ggml_tensor  * b,
      //         struct ggml_tensor  * c);
      m.def(
          "ggml_get_rows_back", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_tensor *b, ggml_tensor *c)
          { return ggml_get_rows_back(ctx_p.ctx, a, b, c); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("b"), nb::arg("c"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_diag(
      //     struct ggml_context     * ctx,
      //     struct ggml_tensor      * a);
      m.def(
          "ggml_diag", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_diag(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // // set elements above the diagonal to -INF
      // GGML_API struct ggml_tensor * ggml_diag_mask_inf(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         int                   n_past);
      m.def(
          "ggml_diag_mask_inf", [](ggml_context_p ctx_p, ggml_tensor *a, int n_past)
          { return ggml_diag_mask_inf(ctx_p.ctx, a, n_past); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("n_past"), nb::rv_policy::reference);

      // // in-place, returns view(a)
      // GGML_API struct ggml_tensor * ggml_diag_mask_inf_inplace(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         int                   n_past);
      m.def(
          "ggml_diag_mask_inf_inplace", [](ggml_context_p ctx_p, ggml_tensor *a, int n_past)
          { return ggml_diag_mask_inf_inplace(ctx_p.ctx, a, n_past); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("n_past"), nb::rv_policy::reference);

      // // set elements above the diagonal to 0
      // GGML_API struct ggml_tensor * ggml_diag_mask_zero(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         int                   n_past);
      m.def(
          "ggml_diag_mask_zero", [](ggml_context_p ctx_p, ggml_tensor *a, int n_past)
          { return ggml_diag_mask_zero(ctx_p.ctx, a, n_past); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("n_past"), nb::rv_policy::reference);

      // // in-place, returns view(a)
      // GGML_API struct ggml_tensor * ggml_diag_mask_zero_inplace(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         int                   n_past);
      m.def(
          "ggml_diag_mask_zero_inplace", [](ggml_context_p ctx_p, ggml_tensor *a, int n_past)
          { return ggml_diag_mask_zero_inplace(ctx_p.ctx, a, n_past); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("n_past"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_soft_max(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_soft_max", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_soft_max(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // // in-place, returns view(a)
      // GGML_API struct ggml_tensor * ggml_soft_max_inplace(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a);
      m.def(
          "ggml_soft_max_inplace", [](ggml_context_p ctx_p, ggml_tensor *a)
          { return ggml_soft_max_inplace(ctx_p.ctx, a); },
          nb::arg("ctx"), nb::arg("a"), nb::rv_policy::reference);

      // // rotary position embedding
      // // if mode & 1 == 1, skip n_past elements
      // // if mode & 2 == 1, GPT-NeoX style
      // // TODO: avoid creating a new tensor every time
      // GGML_API struct ggml_tensor * ggml_rope(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         int                   n_past,
      //         int                   n_dims,
      //         int                   mode);
      m.def(
          "ggml_rope", [](ggml_context_p ctx_p, ggml_tensor *a, int n_past, int n_dims, int mode)
          { return ggml_rope(ctx_p.ctx, a, n_past, n_dims, mode); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("n_past"), nb::arg("n_dims"), nb::arg("mode"), nb::rv_policy::reference);

      // // in-place, returns view(a)
      // GGML_API struct ggml_tensor * ggml_rope_inplace(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         int                   n_past,
      //         int                   n_dims,
      //         int                   mode);
      m.def(
          "ggml_rope_inplace", [](ggml_context_p ctx_p, ggml_tensor *a, int n_past, int n_dims, int mode)
          { return ggml_rope_inplace(ctx_p.ctx, a, n_past, n_dims, mode); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("n_past"), nb::arg("n_dims"), nb::arg("mode"), nb::rv_policy::reference);

      // // rotary position embedding backward, i.e compute dx from dy
      // // a - dy
      // GGML_API struct ggml_tensor * ggml_rope_back(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         int                   n_past,
      //         int                   n_dims,
      //         int                   mode);
      m.def(
          "ggml_rope_back", [](ggml_context_p ctx_p, ggml_tensor *a, int n_past, int n_dims, int mode)
          { return ggml_rope_back(ctx_p.ctx, a, n_past, n_dims, mode); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("n_past"), nb::arg("n_dims"), nb::arg("mode"), nb::rv_policy::reference);

      // // alibi position embedding
      // // in-place, returns view(a)
      // struct ggml_tensor * ggml_alibi(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         int                   n_past,
      //         int                   n_head,
      //         float                 bias_max);
      m.def(
          "ggml_alibi", [](ggml_context_p ctx_p, ggml_tensor *a, int n_past, int n_head, float bias_max)
          { return ggml_alibi(ctx_p.ctx, a, n_past, n_head, bias_max); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("n_past"), nb::arg("n_head"), nb::arg("bias_max"), nb::rv_policy::reference);

      // // clamp
      // // in-place, returns view(a)
      // struct ggml_tensor * ggml_clamp(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         float                 min,
      //         float                 max);
      m.def(
          "ggml_clamp", [](ggml_context_p ctx_p, ggml_tensor *a, float min, float max)
          { return ggml_clamp(ctx_p.ctx, a, min, max); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("min"), nb::arg("max"), nb::rv_policy::reference);

      // // TODO: implement general-purpose convolutions
      // // GGML_API struct ggml_tensor * ggml_conv_1d(
      // //        struct ggml_context * ctx,
      // //        struct ggml_tensor  * a,
      // //        struct ggml_tensor  * b,
      // //        int                   s0
      // //        int                   p0,
      // //        int                   d0);
      // //
      // // GGML_API struct ggml_tensor * ggml_conv_2d(
      // //        struct ggml_context * ctx,
      // //        struct ggml_tensor  * a,
      // //        struct ggml_tensor  * b,
      // //        int                   s0,
      // //        int                   s1,
      // //        int                   p0,
      // //        int                   p1,
      // //        int                   d0,
      // //        int                   d1);

      // // padding = half
      // // TODO: we don't support extra parameters for now
      // //       that's why we are hard-coding the stride, padding, and dilation
      // //       not great ..
      // // example:
      // // a:      3   80  768    1
      // // b:   3000   80    1    1
      // // res: 3000  768    1    1
      // // used in whisper
      // GGML_API struct ggml_tensor * ggml_conv_1d_s1_ph(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         struct ggml_tensor  * b);
      m.def(
          "ggml_conv_1d_s1_ph", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_tensor *b)
          { return ggml_conv_1d_s1_ph(ctx_p.ctx, a, b); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("b"), nb::rv_policy::reference);

      // // used in whisper
      // GGML_API struct ggml_tensor * ggml_conv_1d_s2_ph(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         struct ggml_tensor  * b);
      m.def(
          "ggml_conv_1d_s2_ph", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_tensor *b)
          { return ggml_conv_1d_s2_ph(ctx_p.ctx, a, b); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("b"), nb::rv_policy::reference);

      // // kernel size is a->ne[0] x a->ne[1]
      // // stride is equal to kernel size
      // // padding is zero
      // // example:
      // // a:     16   16    3  768
      // // b:   1024 1024    3    1
      // // res:   64   64  768    1
      // // used in sam
      // GGML_API struct ggml_tensor * ggml_conv_2d_sk_p0(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         struct ggml_tensor  * b);
      m.def(
          "ggml_conv_2d_sk_p0", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_tensor *b)
          { return ggml_conv_2d_sk_p0(ctx_p.ctx, a, b); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("b"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_flash_attn(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * q,
      //         struct ggml_tensor  * k,
      //         struct ggml_tensor  * v,
      //         bool                  masked);
      m.def(
          "ggml_flash_attn", [](ggml_context_p ctx_p, ggml_tensor *q, ggml_tensor *k, ggml_tensor *v, bool masked)
          { return ggml_flash_attn(ctx_p.ctx, q, k, v, masked); },
          nb::arg("ctx"), nb::arg("q"), nb::arg("k"), nb::arg("v"), nb::arg("masked"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_flash_ff(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         struct ggml_tensor  * b0,
      //         struct ggml_tensor  * b1,
      //         struct ggml_tensor  * c0,
      //         struct ggml_tensor  * c1);
      m.def(
          "ggml_flash_ff", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_tensor *b0, ggml_tensor *b1, ggml_tensor *c0, ggml_tensor *c1)
          { return ggml_flash_ff(ctx_p.ctx, a, b0, b1, c0, c1); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("b0"), nb::arg("b1"), nb::arg("c0"), nb::arg("c1"), nb::rv_policy::reference);

      // // partition into non-overlapping windows with padding if needed
      // // example:
      // // a:   768   64   64    1
      // // w:    14
      // // res: 768   14   14    25
      // // used in sam
      // GGML_API struct ggml_tensor * ggml_win_part(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         int                   w);
      m.def(
          "ggml_win_part", [](ggml_context_p ctx_p, ggml_tensor *a, int w)
          { return ggml_win_part(ctx_p.ctx, a, w); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("w"), nb::rv_policy::reference);

      // // reverse of ggml_win_part
      // // used in sam
      // GGML_API struct ggml_tensor * ggml_win_unpart(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor  * a,
      //         int                   w0,
      //         int                   h0,
      //         int                   w);
      m.def(
          "ggml_win_unpart", [](ggml_context_p ctx_p, ggml_tensor *a, int w0, int h0, int w)
          { return ggml_win_unpart(ctx_p.ctx, a, w0, h0, w); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("w0"), nb::arg("h0"), nb::arg("w"), nb::rv_policy::reference);

      // // Mapping operations
      // typedef void (*ggml_unary_op_f32_t)(const int, float *, const float *);
      // typedef void (*ggml_binary_op_f32_t)(const int, float *, const float *, const float *);

      // GGML_API struct ggml_tensor * ggml_map_unary_f32(
      //         struct ggml_context        * ctx,
      //         struct ggml_tensor         * a,
      //                ggml_unary_op_f32_t   fun);
      m.def(
          "ggml_map_unary_f32", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_unary_op_f32_t fun)
          { return ggml_map_unary_f32(ctx_p.ctx, a, fun); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("fun"), nb::rv_policy::reference);

      // GGML_API struct ggml_tensor * ggml_map_binary_f32(
      //         struct ggml_context         * ctx,
      //         struct ggml_tensor          * a,
      //         struct ggml_tensor          * b,
      //                ggml_binary_op_f32_t   fun);
      m.def(
          "ggml_map_binary_f32", [](ggml_context_p ctx_p, ggml_tensor *a, ggml_tensor *b, ggml_binary_op_f32_t fun)
          { return ggml_map_binary_f32(ctx_p.ctx, a, b, fun); },
          nb::arg("ctx"), nb::arg("a"), nb::arg("b"), nb::arg("fun"), nb::rv_policy::reference);

      // //
      // // automatic differentiation
      // //

      // GGML_API void ggml_set_param(
      //         struct ggml_context * ctx,
      //         struct ggml_tensor * tensor);
      m.def(
          "ggml_set_param", [](ggml_context_p ctx_p, ggml_tensor *tensor)
          { return ggml_set_param(ctx_p.ctx, tensor); },
          nb::arg("ctx"), nb::arg("tensor"));

      // GGML_API void ggml_build_forward_expand(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor);
      m.def("ggml_build_forward_expand", &ggml_build_forward_expand, nb::arg("cgraph"), nb::arg("tensor"));

      // GGML_API struct ggml_cgraph ggml_build_forward (struct ggml_tensor * tensor);
      m.def("ggml_build_forward", &ggml_build_forward, nb::arg("tensor"));
      // GGML_API struct ggml_cgraph ggml_build_backward(struct ggml_context * ctx, struct ggml_cgraph * gf, bool keep);
      m.def(
          "ggml_build_backward", [](ggml_context_p ctx_p, ggml_cgraph *gf, bool keep)
          { return ggml_build_backward(ctx_p.ctx, gf, keep); },
          nb::arg("ctx"), nb::arg("gf"), nb::arg("keep"));

      // GGML_API void ggml_graph_compute(struct ggml_context * ctx, struct ggml_cgraph * cgraph);
      m.def(
          "ggml_graph_compute", [](ggml_context_p ctx_p, ggml_cgraph *cgraph)
          { return ggml_graph_compute(ctx_p.ctx, cgraph); },
          nb::arg("ctx"), nb::arg("cgraph"));
      // GGML_API void ggml_graph_reset  (struct ggml_cgraph * cgraph);
      m.def("ggml_graph_reset", &ggml_graph_reset, nb::arg("cgraph"));

      // GGML_API struct ggml_tensor * ggml_graph_get_tensor(struct ggml_cgraph * cgraph, const char * name);
      m.def("ggml_graph_get_tensor", &ggml_graph_get_tensor, nb::arg("cgraph"), nb::arg("name"), nb::rv_policy::reference);

      // GGML_API void               ggml_graph_export(const struct ggml_cgraph * cgraph, const char * fname);
      m.def("ggml_graph_export", &ggml_graph_export, nb::arg("cgraph"), nb::arg("fname"));
      // GGML_API struct ggml_cgraph ggml_graph_import(const char * fname, struct ggml_context ** ctx_data, struct ggml_context ** ctx_eval);
      m.def(
          "ggml_graph_import",
          [](const char *fname, ggml_context_p ctx_data, ggml_context_p ctx_eval)
          {
                return ggml_graph_import(fname, &ctx_data.ctx, &ctx_eval.ctx);
          },
          nb::arg("fname"), nb::arg("ctx_data"), nb::arg("ctx_eval"));

      // // print info and performance information for the graph
      // GGML_API void ggml_graph_print(const struct ggml_cgraph * cgraph);
      m.def("ggml_graph_print", &ggml_graph_print, nb::arg("cgraph"));

      // // dump the graph into a file using the dot format
      // GGML_API void ggml_graph_dump_dot(const struct ggml_cgraph * gb, const struct ggml_cgraph * gf, const char * filename);
      m.def("ggml_graph_dump_dot", &ggml_graph_dump_dot, nb::arg("gb"), nb::arg("gf"), nb::arg("filename"));

      // //
      // // optimization
      // //

      // // optimization methods
      // enum ggml_opt_type {
      //     GGML_OPT_ADAM,
      //     GGML_OPT_LBFGS,
      // };
      nb::enum_<ggml_opt_type>(m, "ggml_opt_type")
          .value("GGML_OPT_ADAM", ggml_opt_type::GGML_OPT_ADAM)
          .value("GGML_OPT_LBFGS", ggml_opt_type::GGML_OPT_LBFGS)
          .export_values();

      // // linesearch methods
      // enum ggml_linesearch {
      //     GGML_LINESEARCH_DEFAULT = 1,

      //     GGML_LINESEARCH_BACKTRACKING_ARMIJO       = 0,
      //     GGML_LINESEARCH_BACKTRACKING_WOLFE        = 1,
      //     GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 2,
      // };
      nb::enum_<ggml_linesearch>(m, "ggml_linesearch")
          .value("GGML_LINESEARCH_DEFAULT", ggml_linesearch::GGML_LINESEARCH_DEFAULT)
          .value("GGML_LINESEARCH_BACKTRACKING_ARMIJO", ggml_linesearch::GGML_LINESEARCH_BACKTRACKING_ARMIJO)
          .value("GGML_LINESEARCH_BACKTRACKING_WOLFE", ggml_linesearch::GGML_LINESEARCH_BACKTRACKING_WOLFE)
          .value("GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE", ggml_linesearch::GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE)
          .export_values();

      // // optimization return values
      // enum ggml_opt_result {
      //     GGML_OPT_OK = 0,
      //     GGML_OPT_DID_NOT_CONVERGE,
      //     GGML_OPT_NO_CONTEXT,
      //     GGML_OPT_INVALID_WOLFE,
      //     GGML_OPT_FAIL,

      //     GGML_LINESEARCH_FAIL = -128,
      //     GGML_LINESEARCH_MINIMUM_STEP,
      //     GGML_LINESEARCH_MAXIMUM_STEP,
      //     GGML_LINESEARCH_MAXIMUM_ITERATIONS,
      //     GGML_LINESEARCH_INVALID_PARAMETERS,
      // };
      nb::enum_<ggml_opt_result>(m, "ggml_opt_result")
          .value("GGML_OPT_OK", ggml_opt_result::GGML_OPT_OK)
          .value("GGML_OPT_DID_NOT_CONVERGE", ggml_opt_result::GGML_OPT_DID_NOT_CONVERGE)
          .value("GGML_OPT_NO_CONTEXT", ggml_opt_result::GGML_OPT_NO_CONTEXT)
          .value("GGML_OPT_INVALID_WOLFE", ggml_opt_result::GGML_OPT_INVALID_WOLFE)
          .value("GGML_OPT_FAIL", ggml_opt_result::GGML_OPT_FAIL)
          .value("GGML_LINESEARCH_FAIL", ggml_opt_result::GGML_LINESEARCH_FAIL)
          .value("GGML_LINESEARCH_MINIMUM_STEP", ggml_opt_result::GGML_LINESEARCH_MINIMUM_STEP)
          .value("GGML_LINESEARCH_MAXIMUM_STEP", ggml_opt_result::GGML_LINESEARCH_MAXIMUM_STEP)
          .value("GGML_LINESEARCH_MAXIMUM_ITERATIONS", ggml_opt_result::GGML_LINESEARCH_MAXIMUM_ITERATIONS)
          .value("GGML_LINESEARCH_INVALID_PARAMETERS", ggml_opt_result::GGML_LINESEARCH_INVALID_PARAMETERS)
          .export_values();

      // // optimization parameters
      // //
      // //   see ggml.c (ggml_opt_default_params) for default values
      // //
      // struct ggml_opt_params {
      //     enum ggml_opt_type type;

      //     int n_threads;

      //     // delta-based convergence test
      //     //
      //     //   if past == 0 - disabled
      //     //   if past > 0:
      //     //     stop if |f(x) - f(x_past)| < delta * max(1, |f(x)|)
      //     //
      //     int past;
      //     float delta;

      //     // maximum number of iterations without improvement
      //     //
      //     //   if 0 - disabled
      //     //   if > 0:
      //     //     assume convergence if no cost improvement in this number of iterations
      //     //
      //     int max_no_improvement;

      //     bool print_forward_graph;
      //     bool print_backward_graph;

      //     // ADAM parameters
      //     struct {
      //         int n_iter;

      //         float alpha; // learning rate
      //         float beta1;
      //         float beta2;
      //         float eps;   // epsilon for numerical stability
      //         float eps_f; // epsilon for convergence test
      //         float eps_g; // epsilon for convergence test
      //     } adam;

      //     // LBFGS parameters
      //     struct {
      //         int m; // number of corrections to approximate the inv. Hessian
      //         int n_iter;
      //         int max_linesearch;

      //         float eps;      // convergence tolerance
      //         float ftol;     // line search tolerance
      //         float wolfe;
      //         float min_step;
      //         float max_step;

      //         enum ggml_linesearch linesearch;
      //     } lbfgs;
      // };
      nb::class_<ggml_opt_params>(m, "ggml_opt_params");

      // GGML_API struct ggml_opt_params ggml_opt_default_params(enum ggml_opt_type type);
      m.def("ggml_opt_default_params", &ggml_opt_default_params, nb::arg("type"));

      // // optimize the function defined by the tensor f
      // GGML_API enum ggml_opt_result ggml_opt(
      //         struct ggml_context * ctx,
      //         struct ggml_opt_params params,
      //         struct ggml_tensor * f);
      m.def(
          "ggml_opt", [](ggml_context_p ctx_p, ggml_opt_params params, ggml_tensor *f)
          { return ggml_opt(ctx_p.ctx, params, f); },
          nb::arg("ctx"), nb::arg("params"), nb::arg("f"));

      // //
      // // quantization
      // //

      // GGML_API size_t ggml_quantize_q4_0(const float * src, void * dst, int n, int k, int64_t * hist);
      m.def("ggml_quantize_q4_0", &ggml_quantize_q4_0, nb::arg("src"), nb::arg("dst"), nb::arg("n"), nb::arg("k"), nb::arg("hist"));
      // GGML_API size_t ggml_quantize_q4_1(const float * src, void * dst, int n, int k, int64_t * hist);
      m.def("ggml_quantize_q4_1", &ggml_quantize_q4_1, nb::arg("src"), nb::arg("dst"), nb::arg("n"), nb::arg("k"), nb::arg("hist"));
      // GGML_API size_t ggml_quantize_q5_0(const float * src, void * dst, int n, int k, int64_t * hist);
      m.def("ggml_quantize_q5_0", &ggml_quantize_q5_0, nb::arg("src"), nb::arg("dst"), nb::arg("n"), nb::arg("k"), nb::arg("hist"));
      // GGML_API size_t ggml_quantize_q5_1(const float * src, void * dst, int n, int k, int64_t * hist);
      m.def("ggml_quantize_q5_1", &ggml_quantize_q5_1, nb::arg("src"), nb::arg("dst"), nb::arg("n"), nb::arg("k"), nb::arg("hist"));
      // GGML_API size_t ggml_quantize_q8_0(const float * src, void * dst, int n, int k, int64_t * hist);
      m.def("ggml_quantize_q8_0", &ggml_quantize_q8_0, nb::arg("src"), nb::arg("dst"), nb::arg("n"), nb::arg("k"), nb::arg("hist"));

      // GGML_API size_t ggml_quantize_chunk(enum ggml_type type, const float * src, void * dst, int start, int n, int64_t * hist);
      m.def("ggml_quantize_chunk", &ggml_quantize_chunk, nb::arg("type"), nb::arg("src"), nb::arg("dst"), nb::arg("start"), nb::arg("n"), nb::arg("hist"));

      // //
      // // system info
      // //

      // GGML_API int ggml_cpu_has_avx        (void);
      m.def("ggml_cpu_has_avx", &ggml_cpu_has_avx);
      // GGML_API int ggml_cpu_has_avx2       (void);
      m.def("ggml_cpu_has_avx2", &ggml_cpu_has_avx2);
      // GGML_API int ggml_cpu_has_avx512     (void);
      m.def("ggml_cpu_has_avx512", &ggml_cpu_has_avx512);
      // GGML_API int ggml_cpu_has_avx512_vbmi(void);
      m.def("ggml_cpu_has_avx512_vbmi", &ggml_cpu_has_avx512_vbmi);
      // GGML_API int ggml_cpu_has_avx512_vnni(void);
      m.def("ggml_cpu_has_avx512_vnni", &ggml_cpu_has_avx512_vnni);
      // GGML_API int ggml_cpu_has_fma        (void);
      m.def("ggml_cpu_has_fma", &ggml_cpu_has_fma);
      // GGML_API int ggml_cpu_has_neon       (void);
      m.def("ggml_cpu_has_neon", &ggml_cpu_has_neon);
      // GGML_API int ggml_cpu_has_arm_fma    (void);
      m.def("ggml_cpu_has_arm_fma", &ggml_cpu_has_arm_fma);
      // GGML_API int ggml_cpu_has_f16c       (void);
      m.def("ggml_cpu_has_f16c", &ggml_cpu_has_f16c);
      // GGML_API int ggml_cpu_has_fp16_va    (void);
      m.def("ggml_cpu_has_fp16_va", &ggml_cpu_has_fp16_va);
      // GGML_API int ggml_cpu_has_wasm_simd  (void);
      m.def("ggml_cpu_has_wasm_simd", &ggml_cpu_has_wasm_simd);
      // GGML_API int ggml_cpu_has_blas       (void);
      m.def("ggml_cpu_has_blas", &ggml_cpu_has_blas);
      // GGML_API int ggml_cpu_has_cublas     (void);
      m.def("ggml_cpu_has_cublas", &ggml_cpu_has_cublas);
      // GGML_API int ggml_cpu_has_clblast    (void);
      m.def("ggml_cpu_has_clblast", &ggml_cpu_has_clblast);
      // GGML_API int ggml_cpu_has_gpublas    (void);
      m.def("ggml_cpu_has_gpublas", &ggml_cpu_has_gpublas);
      // GGML_API int ggml_cpu_has_sse3       (void);
      m.def("ggml_cpu_has_sse3", &ggml_cpu_has_sse3);
      // GGML_API int ggml_cpu_has_vsx        (void);
      m.def("ggml_cpu_has_vsx", &ggml_cpu_has_vsx);
}
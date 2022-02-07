//
// [PR描述](https://github.com/pybind/pybind11/pull/3544)
// 不确定PR是否会被合并, 如果合并了, 这个feature补丁将会被删除.
// 当前对应的版本为 pybind11 2.9.0
// 由于结构体命名冲突
// `npy_api` -> `npy_api_patch`
// `is_complex` -> `is_complex_patch`
// `npy_format_descriptor_name` -> `npy_format_descriptor_name_patch`
//
// Created by 孙瑞琦 on 2021/12/25.
//
//

#pragma once

#include "pybind11/pybind11.h"
#include "pybind11/complex.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

template<typename> struct numpy_scalar; // Forward declaration

PYBIND11_NAMESPACE_BEGIN(detail)

template <std::size_t> constexpr int platform_lookup() { return -1; }

// Lookup a type according to its size, and return a value corresponding to the NumPy typenum.
template <std::size_t size, typename T, typename... Ts, typename... Ints>
constexpr int platform_lookup(int I, Ints... Is) {
    return sizeof(size) == sizeof(T) ? I : platform_lookup<size, Ts...>(Is...);
}

struct npy_api_patch {
    enum constants {
        NPY_ARRAY_C_CONTIGUOUS_ = 0x0001,
        NPY_ARRAY_F_CONTIGUOUS_ = 0x0002,
        NPY_ARRAY_OWNDATA_ = 0x0004,
        NPY_ARRAY_FORCECAST_ = 0x0010,
        NPY_ARRAY_ENSUREARRAY_ = 0x0040,
        NPY_ARRAY_ALIGNED_ = 0x0100,
        NPY_ARRAY_WRITEABLE_ = 0x0400,
        NPY_BOOL_ = 0,
        NPY_BYTE_, NPY_UBYTE_,
        NPY_SHORT_, NPY_USHORT_,
        NPY_INT_, NPY_UINT_,
        NPY_LONG_, NPY_ULONG_,
        NPY_LONGLONG_, NPY_ULONGLONG_,
        NPY_FLOAT_, NPY_DOUBLE_, NPY_LONGDOUBLE_,
        NPY_CFLOAT_, NPY_CDOUBLE_, NPY_CLONGDOUBLE_,
        NPY_OBJECT_ = 17,
        NPY_STRING_, NPY_UNICODE_, NPY_VOID_,
        // Platform-dependent normalization
        NPY_INT8_ = NPY_BYTE_,
        NPY_UINT8_ = NPY_UBYTE_,
        NPY_INT16_ = NPY_SHORT_,
        NPY_UINT16_ = NPY_USHORT_,
        // `npy_common.h` defines the integer aliases. In order, it checks:
        // NPY_BITSOF_LONG, NPY_BITSOF_LONGLONG, NPY_BITSOF_INT, NPY_BITSOF_SHORT, NPY_BITSOF_CHAR
        // and assigns the alias to the first matching size, so we should check in this order.
        NPY_INT32_ = platform_lookup<4, long, int, short>(
                NPY_LONG_, NPY_INT_, NPY_SHORT_),
        NPY_UINT32_ = platform_lookup<4, unsigned long, unsigned int, unsigned short>(
                NPY_ULONG_, NPY_UINT_, NPY_USHORT_),
        NPY_INT64_ = platform_lookup<8, long, long long, int>(
                NPY_LONG_, NPY_LONGLONG_, NPY_INT_),
        NPY_UINT64_ = platform_lookup<8, unsigned long, unsigned long long, unsigned int>(
                NPY_ULONG_, NPY_ULONGLONG_, NPY_UINT_),
        NPY_FLOAT32_ = platform_lookup<4, double, float, long double>(
                NPY_DOUBLE_, NPY_FLOAT_, NPY_LONGDOUBLE_),
        NPY_FLOAT64_ = platform_lookup<8, double, float, long double>(
                NPY_DOUBLE_, NPY_FLOAT_, NPY_LONGDOUBLE_),
        NPY_COMPLEX64_ = platform_lookup<8, std::complex<double>, std::complex<float>, std::complex<long double>>(
                NPY_DOUBLE_, NPY_FLOAT_, NPY_LONGDOUBLE_),
        NPY_COMPLEX128_ = platform_lookup<8, std::complex<double>, std::complex<float>, std::complex<long double>>(
                NPY_DOUBLE_, NPY_FLOAT_, NPY_LONGDOUBLE_),
        NPY_CHAR_ = std::is_signed<char>::value ? NPY_BYTE_ : NPY_UBYTE_,
    };

    struct PyArray_Dims {
        Py_intptr_t *ptr;
        int len;
    };

    static npy_api_patch& get() {
        static npy_api_patch api = lookup();
        return api;
    }

    bool PyArray_Check_(PyObject *obj) const {
        return PyObject_TypeCheck(obj, PyArray_Type_) != 0;
    }
    bool PyArrayDescr_Check_(PyObject *obj) const {
        return PyObject_TypeCheck(obj, PyArrayDescr_Type_) != 0;
    }

    unsigned int (*PyArray_GetNDArrayCFeatureVersion_)();
    PyObject *(*PyArray_DescrFromType_)(int);
    PyObject *(*PyArray_TypeObjectFromType_)(int);
    PyObject *(*PyArray_NewFromDescr_)
            (PyTypeObject *, PyObject *, int, Py_intptr_t const *,
             Py_intptr_t const *, void *, int, PyObject *);
    // Unused. Not removed because that affects ABI of the class.
    PyObject *(*PyArray_DescrNewFromType_)(int);
    int (*PyArray_CopyInto_)(PyObject *, PyObject *);
    PyObject *(*PyArray_NewCopy_)(PyObject *, int);
    PyTypeObject *PyArray_Type_;
    PyTypeObject *PyVoidArrType_Type_;
    PyTypeObject *PyArrayDescr_Type_;
    PyObject *(*PyArray_DescrFromScalar_)(PyObject *);
    PyObject *(*PyArray_Scalar_)(void *, PyObject *, PyObject *);
    void (*PyArray_ScalarAsCtype_)(PyObject *, void *);
    PyObject *(*PyArray_FromAny_) (PyObject *, PyObject *, int, int, int, PyObject *);
    int (*PyArray_DescrConverter_) (PyObject *, PyObject **);
    bool (*PyArray_EquivTypes_) (PyObject *, PyObject *);
    int (*PyArray_GetArrayParamsFromObject_)(PyObject *, PyObject *, unsigned char, PyObject **, int *,
                                             Py_intptr_t *, PyObject **, PyObject *);
    PyObject *(*PyArray_Squeeze_)(PyObject *);
    // Unused. Not removed because that affects ABI of the class.
    int (*PyArray_SetBaseObject_)(PyObject *, PyObject *);
    PyObject* (*PyArray_Resize_)(PyObject*, PyArray_Dims*, int, int);
    PyObject* (*PyArray_Newshape_)(PyObject*, PyArray_Dims*, int);
    PyObject* (*PyArray_View_)(PyObject*, PyObject*, PyObject*);

private:
    enum functions {
        API_PyArray_GetNDArrayCFeatureVersion = 211,
        API_PyArray_Type = 2,
        API_PyArrayDescr_Type = 3,
        API_PyVoidArrType_Type = 39,
        API_PyArray_DescrFromType = 45,
        API_PyArray_TypeObjectFromType = 46,
        API_PyArray_DescrFromScalar = 57,
        API_PyArray_Scalar = 60,
        API_PyArray_ScalarAsCtype = 62,
        API_PyArray_FromAny = 69,
        API_PyArray_Resize = 80,
        API_PyArray_CopyInto = 82,
        API_PyArray_NewCopy = 85,
        API_PyArray_NewFromDescr = 94,
        API_PyArray_DescrNewFromType = 96,
        API_PyArray_Newshape = 135,
        API_PyArray_Squeeze = 136,
        API_PyArray_View = 137,
        API_PyArray_DescrConverter = 174,
        API_PyArray_EquivTypes = 182,
        API_PyArray_GetArrayParamsFromObject = 278,
        API_PyArray_SetBaseObject = 282
    };

    static npy_api_patch lookup() {
        module_ m = module_::import("numpy.core.multiarray");
        auto c = m.attr("_ARRAY_API");
#if PY_MAJOR_VERSION >= 3
        void **api_ptr = (void **) PyCapsule_GetPointer(c.ptr(), NULL);
#else
        void **api_ptr = (void **) PyCObject_AsVoidPtr(c.ptr());
#endif
        npy_api_patch api;
#define DECL_NPY_API(Func) api.Func##_ = (decltype(api.Func##_)) api_ptr[API_##Func];
        DECL_NPY_API(PyArray_GetNDArrayCFeatureVersion);
        if (api.PyArray_GetNDArrayCFeatureVersion_() < 0x7)
            pybind11_fail("pybind11 numpy support requires numpy >= 1.7.0");
        DECL_NPY_API(PyArray_Type);
        DECL_NPY_API(PyVoidArrType_Type);
        DECL_NPY_API(PyArrayDescr_Type);
        DECL_NPY_API(PyArray_DescrFromType);
        DECL_NPY_API(PyArray_TypeObjectFromType);
        DECL_NPY_API(PyArray_DescrFromScalar);
        DECL_NPY_API(PyArray_Scalar);
        DECL_NPY_API(PyArray_ScalarAsCtype);
        DECL_NPY_API(PyArray_FromAny);
        DECL_NPY_API(PyArray_Resize);
        DECL_NPY_API(PyArray_CopyInto);
        DECL_NPY_API(PyArray_NewCopy);
        DECL_NPY_API(PyArray_NewFromDescr);
        DECL_NPY_API(PyArray_DescrNewFromType);
        DECL_NPY_API(PyArray_Newshape);
        DECL_NPY_API(PyArray_Squeeze);
        DECL_NPY_API(PyArray_View);
        DECL_NPY_API(PyArray_DescrConverter);
        DECL_NPY_API(PyArray_EquivTypes);
        DECL_NPY_API(PyArray_GetArrayParamsFromObject);
        DECL_NPY_API(PyArray_SetBaseObject);

#undef DECL_NPY_API
        return api;
    }
};

template <typename T> struct is_complex_patch : std::false_type { };
template <typename T> struct is_complex_patch<std::complex<T>> : std::true_type { };

template <typename T, typename = void>
struct npy_format_descriptor_name_patch;

template <typename T>
struct npy_format_descriptor_name_patch<T, enable_if_t<std::is_integral<T>::value>> {
    static constexpr auto name = const_name<std::is_same<T, bool>::value>(
        const_name("bool"), const_name<std::is_signed<T>::value>("int", "uint") + const_name<sizeof(T)*8>()
    );
};

template <typename T>
struct npy_format_descriptor_name_patch<T, enable_if_t<std::is_floating_point<T>::value>> {
    static constexpr auto name = const_name<std::is_same<T, float>::value || std::is_same<T, double>::value>(
        const_name("float") + const_name<sizeof(T)*8>(), const_name("longdouble")
    );
};

template <typename T>
struct npy_format_descriptor_name_patch<T, enable_if_t<is_complex_patch<T>::value>> {
    static constexpr auto name = const_name<std::is_same<typename T::value_type, float>::value
                               || std::is_same<typename T::value_type, double>::value>(
        const_name("complex") + const_name<sizeof(typename T::value_type)*16>(), const_name("longcomplex")
    );
};

template<typename T> struct numpy_scalar_info {};

#define DECL_NPY_SCALAR(ctype_, typenum_) \
template<> struct numpy_scalar_info<ctype_> { \
    static constexpr auto name = npy_format_descriptor_name_patch<ctype_>::name; \
    static constexpr int typenum = npy_api_patch::typenum_##_; \
}

// boolean type
DECL_NPY_SCALAR(bool, NPY_BOOL);

// character types
DECL_NPY_SCALAR(char, NPY_CHAR);
DECL_NPY_SCALAR(signed char, NPY_BYTE);
DECL_NPY_SCALAR(unsigned char, NPY_UBYTE);

// signed integer types
DECL_NPY_SCALAR(std::int16_t, NPY_SHORT);
DECL_NPY_SCALAR(std::int32_t, NPY_INT);
DECL_NPY_SCALAR(std::int64_t, NPY_LONG);
#if defined(__linux__)
DECL_NPY_SCALAR(long long, NPY_LONG);
#else
DECL_NPY_SCALAR(long, NPY_LONG);
#endif

// unsigned integer types
DECL_NPY_SCALAR(std::uint16_t, NPY_USHORT);
DECL_NPY_SCALAR(std::uint32_t, NPY_UINT);
DECL_NPY_SCALAR(std::uint64_t, NPY_ULONG);
#if defined(__linux__)
DECL_NPY_SCALAR(unsigned long long, NPY_ULONG);
#else
DECL_NPY_SCALAR(unsigned long, NPY_ULONG);
#endif

// floating point types
DECL_NPY_SCALAR(float, NPY_FLOAT);
DECL_NPY_SCALAR(double, NPY_DOUBLE);
DECL_NPY_SCALAR(long double, NPY_LONGDOUBLE);

// complex types
DECL_NPY_SCALAR(std::complex<float>, NPY_CFLOAT);
DECL_NPY_SCALAR(std::complex<double>, NPY_CDOUBLE);
DECL_NPY_SCALAR(std::complex<long double>, NPY_CLONGDOUBLE);

#undef DECL_NPY_SCALAR

template<typename T>
struct type_caster<numpy_scalar<T>> {
    using value_type = T;
    using type_info = numpy_scalar_info<T>;

    PYBIND11_TYPE_CASTER(numpy_scalar<T>, type_info::name);

    static handle& target_type() {
        static handle tp = npy_api_patch::get().PyArray_TypeObjectFromType_(type_info::typenum);
        return tp;
    }

    static handle& target_dtype() {
        static handle tp = npy_api_patch::get().PyArray_DescrFromType_(type_info::typenum);
        return tp;
    }

    bool load(handle src, bool) {
        if (isinstance(src, target_type())) {
            npy_api_patch::get().PyArray_ScalarAsCtype_(src.ptr(), &value.value);
            return true;
        }
        return false;
    }

    static handle cast(numpy_scalar<T> src, return_value_policy, handle) {
        return npy_api_patch::get().PyArray_Scalar_(&src.value, target_dtype().ptr(), nullptr);
    }
};

PYBIND11_NAMESPACE_END(detail)

template<typename T>
struct numpy_scalar {
    using value_type = T;

    value_type value;

    numpy_scalar() = default;
    numpy_scalar(value_type value) : value(value) {}

    operator value_type() { return value; }
    numpy_scalar& operator=(value_type value) { this->value = value; return *this; }
};

template<typename T>
numpy_scalar<T> make_scalar(T value) {
    return numpy_scalar<T>(value);
}

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
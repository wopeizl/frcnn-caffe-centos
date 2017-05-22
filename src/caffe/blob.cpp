#include <climits>
#include <vector>
#include "fstream"
#include "sstream"
#include "iostream"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
/***********************add for pruning*****************************/
int step = 0;
float thre_fc6 = 0.003;
float thre_fc7 = 0.007;
/*******************************************************************/

template <typename Dtype>
    void Blob<Dtype>::Reshape(Blob<Dtype>& origin) {
        shape_data_.swap(origin.shape_data_);
        data_.swap(origin.data_);
        shape_.swap(origin.shape_);
        capacity_ = origin.capacity_;
        count_ = origin.count_;
        diff_.swap(origin.diff_);
    }

template <typename Dtype>
void Blob<Dtype>::Reshape(const int num, const int channels, const int height,
    const int width) {
  vector<int> shape(4);
  shape[0] = num;
  shape[1] = channels;
  shape[2] = height;
  shape[3] = width;
  Reshape(shape);
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const vector<int>& shape) {
  CHECK_LE(shape.size(), kMaxBlobAxes);
  count_ = 1;
  shape_.resize(shape.size());
  if (!shape_data_ || shape_data_->size() < shape.size() * sizeof(int)) {
    shape_data_.reset(new SyncedMemory(shape.size() * sizeof(int)));
  }
  int* shape_data = static_cast<int*>(shape_data_->mutable_cpu_data());
  for (int i = 0; i < shape.size(); ++i) {
    CHECK_GE(shape[i], 0);
    if (count_ != 0) {
      CHECK_LE(shape[i], INT_MAX / count_) << "blob size exceeds INT_MAX";
    }
    count_ *= shape[i];
    shape_[i] = shape[i];
    shape_data[i] = shape[i];
  }
  if (count_ > capacity_) {
    capacity_ = count_;
    data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
    diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
    /*********************add for pruning********************************/
    mask_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
    /********************************************************************/
  }
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const BlobShape& shape) {
  CHECK_LE(shape.dim_size(), kMaxBlobAxes);
  vector<int> shape_vec(shape.dim_size());
  for (int i = 0; i < shape.dim_size(); ++i) {
    shape_vec[i] = shape.dim(i);
  }
  Reshape(shape_vec);
}

template <typename Dtype>
void Blob<Dtype>::ReshapeLike(const Blob<Dtype>& other) {
  Reshape(other.shape());
}

template <typename Dtype>
Blob<Dtype>::Blob(const int num, const int channels, const int height,
    const int width)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0) {
  Reshape(num, channels, height, width);
}

template <typename Dtype>
Blob<Dtype>::Blob(const vector<int>& shape)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0)
{
    Reshape(shape);
}

template <typename Dtype>
const int* Blob<Dtype>::gpu_shape() const {
  CHECK(shape_data_);
  return (const int*)shape_data_->gpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->cpu_data();
}

template <typename Dtype>
void Blob<Dtype>::set_cpu_data(Dtype* data) {
  CHECK(data);
  // Make sure CPU and GPU sizes remain equal
  size_t size = count_ * sizeof(Dtype);
  if (data_->size() != size) {
    data_.reset(new SyncedMemory(size));
    diff_.reset(new SyncedMemory(size));
  }
  data_->set_cpu_data(data);
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->gpu_data();
}

template <typename Dtype>
void Blob<Dtype>::set_gpu_data(Dtype* data) {
  CHECK(data);
  // Make sure CPU and GPU sizes remain equal
  size_t size = count_ * sizeof(Dtype);
  if (data_->size() != size) {
    data_.reset(new SyncedMemory(size));
    diff_.reset(new SyncedMemory(size));
  }
  data_->set_gpu_data(data);
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_diff() const {
  CHECK(diff_);
  return (const Dtype*)diff_->cpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_diff() const {
  CHECK(diff_);
  return (const Dtype*)diff_->gpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_gpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_diff() {
  CHECK(diff_);
  return static_cast<Dtype*>(diff_->mutable_gpu_data());
}

template <typename Dtype>
void Blob<Dtype>::ShareData(const Blob& other) {
  CHECK_EQ(count_, other.count());
  data_ = other.data();
}

template <typename Dtype>
void Blob<Dtype>::ShareDiff(const Blob& other) {
  CHECK_EQ(count_, other.count());
  diff_ = other.diff();
}
/********************add for pruning**********************/
template <typename Dtype>
void Blob<Dtype>::set_cpu_mask(Dtype* mask) {
  CHECK(mask);
  mask_->set_cpu_data(mask);
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_mask() const {
  CHECK(mask_);
  return (const Dtype*)mask_->cpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_mask() const {
  CHECK(mask_);
  return (const Dtype*)mask_->gpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_mask() {
  CHECK(mask_);
  return static_cast<Dtype*>(mask_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_mask() {
  CHECK(mask_);
  return static_cast<Dtype*>(mask_->mutable_gpu_data());
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_csrval() const {
  CHECK(csrval_);
  return (const Dtype*)csrval_->cpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_csrval() const {
  CHECK(csrval_);
  return (const Dtype*)csrval_->gpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_csrval() {
  CHECK(csrval_);
  return static_cast<Dtype*>(csrval_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_csrval() {
  CHECK(csrval_);
  return static_cast<Dtype*>(csrval_->mutable_gpu_data());
}

template <typename Dtype>
const int * Blob<Dtype>::cpu_csrrowptr() const {
  CHECK(csrrowptr_);
  return (const int *)csrrowptr_->cpu_data();
}

template <typename Dtype>
const int * Blob<Dtype>::gpu_csrrowptr() const {
  CHECK(csrrowptr_);
  return (const int *)csrrowptr_->gpu_data();
}

template <typename Dtype>
int * Blob<Dtype>::mutable_cpu_csrrowptr() {
  CHECK(csrrowptr_);
  return static_cast<int *>(csrrowptr_->mutable_cpu_data());
}

template <typename Dtype>
int * Blob<Dtype>::mutable_gpu_csrrowptr() {
  CHECK(csrrowptr_);
  return static_cast<int *>(csrrowptr_->mutable_gpu_data());
}

template <typename Dtype>
const int * Blob<Dtype>::cpu_csrcolind() const {
  CHECK(csrcolind_);
  return (const int *)csrcolind_->cpu_data();
}

template <typename Dtype>
const int * Blob<Dtype>::gpu_csrcolind() const {
  CHECK(csrcolind_);
  return (const int *)csrcolind_->gpu_data();
}

template <typename Dtype>
int * Blob<Dtype>::mutable_cpu_csrcolind() {
  CHECK(csrcolind_);
  return static_cast<int *>(csrcolind_->mutable_cpu_data());
}

template <typename Dtype>
int * Blob<Dtype>::mutable_gpu_csrcolind() {
  CHECK(csrcolind_);
  return static_cast<int *>(csrcolind_->mutable_gpu_data());
}

/*********************************************************/
// The "update" method is used for parameter blobs in a Net, which are stored
// as Blob<float> or Blob<double> -- hence we do not define it for
// Blob<int> or Blob<unsigned int>.
template <> void Blob<unsigned int>::Update() { NOT_IMPLEMENTED; }
template <> void Blob<int>::Update() { NOT_IMPLEMENTED; }

template <typename Dtype>
void Blob<Dtype>::Update() {
  // We will perform update based on where the data is located.
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    // perform computation on CPU
    caffe_axpy<Dtype>(count_, Dtype(-1),
        static_cast<const Dtype*>(diff_->cpu_data()),
        static_cast<Dtype*>(data_->mutable_cpu_data()));
    /*********************add for pruning*************************/
    if(step==2 && blob_id_==0)
    {
      if(layer_type_==1 && (layer_id_>=1 && layer_id_<=2))
      {
        caffe_mul<Dtype>(count_,
        static_cast<const Dtype*>(mask_->cpu_data()),
        static_cast<const Dtype*>(data_->cpu_data()),
        static_cast<Dtype*>(data_->mutable_cpu_data()));
      }
    }
    /*************************************************************/
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    // perform computation on GPU
    caffe_gpu_axpy<Dtype>(count_, Dtype(-1),
        static_cast<const Dtype*>(diff_->gpu_data()),
        static_cast<Dtype*>(data_->mutable_gpu_data()));
    /*********************add for pruning*************************/
    if(step==2 && blob_id_==0)
    {
      if(layer_type_==1 && (layer_id_>=1 && layer_id_<=2))
      {
        caffe_gpu_mul<Dtype>(count_,
        static_cast<const Dtype*>(mask_->gpu_data()),
        static_cast<const Dtype*>(data_->gpu_data()),
        static_cast<Dtype*>(data_->mutable_gpu_data()));
      }
    }
    /*************************************************************/
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Syncedmem not initialized.";
  }
}

template <> unsigned int Blob<unsigned int>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::asum_data() const {
  if (!data_) { return 0; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    return caffe_cpu_asum(count_, cpu_data());
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
  {
    Dtype asum;
    caffe_gpu_asum(count_, gpu_data(), &asum);
    return asum;
  }
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return 0;
}

template <> unsigned int Blob<unsigned int>::asum_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::asum_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::asum_diff() const {
  if (!diff_) { return 0; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    return caffe_cpu_asum(count_, cpu_diff());
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
  {
    Dtype asum;
    caffe_gpu_asum(count_, gpu_diff(), &asum);
    return asum;
  }
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
  }
  return 0;
}

template <> unsigned int Blob<unsigned int>::sumsq_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::sumsq_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::sumsq_data() const {
  Dtype sumsq;
  const Dtype* data;
  if (!data_) { return 0; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    data = cpu_data();
    sumsq = caffe_cpu_dot(count_, data, data);
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    data = gpu_data();
    caffe_gpu_dot(count_, data, data, &sumsq);
#else
    NO_GPU;
#endif
    break;
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return sumsq;
}

template <> unsigned int Blob<unsigned int>::sumsq_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::sumsq_diff() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
Dtype Blob<Dtype>::sumsq_diff() const {
  Dtype sumsq;
  const Dtype* diff;
  if (!diff_) { return 0; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    diff = cpu_diff();
    sumsq = caffe_cpu_dot(count_, diff, diff);
    break;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    diff = gpu_diff();
    caffe_gpu_dot(count_, diff, diff, &sumsq);
    break;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return 0;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
  return sumsq;
}

template <> void Blob<unsigned int>::scale_data(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template <> void Blob<int>::scale_data(int scale_factor) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Blob<Dtype>::scale_data(Dtype scale_factor) {
  Dtype* data;
  if (!data_) { return; }
  switch (data_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    data = mutable_cpu_data();
    caffe_scal(count_, scale_factor, data);
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    data = mutable_gpu_data();
    caffe_gpu_scal(count_, scale_factor, data);
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
  }
}

template <> void Blob<unsigned int>::scale_diff(unsigned int scale_factor) {
  NOT_IMPLEMENTED;
}

template <> void Blob<int>::scale_diff(int scale_factor) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Blob<Dtype>::scale_diff(Dtype scale_factor) {
  Dtype* diff;
  if (!diff_) { return; }
  switch (diff_->head()) {
  case SyncedMemory::HEAD_AT_CPU:
    diff = mutable_cpu_diff();
    caffe_scal(count_, scale_factor, diff);
    return;
  case SyncedMemory::HEAD_AT_GPU:
  case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
    diff = mutable_gpu_diff();
    caffe_gpu_scal(count_, scale_factor, diff);
    return;
#else
    NO_GPU;
#endif
  case SyncedMemory::UNINITIALIZED:
    return;
  default:
    LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
  }
}

template <typename Dtype>
bool Blob<Dtype>::ShapeEquals(const BlobProto& other) {
  if (other.has_num() || other.has_channels() ||
      other.has_height() || other.has_width()) {
    // Using deprecated 4D Blob dimensions --
    // shape is (num, channels, height, width).
    // Note: we do not use the normal Blob::num(), Blob::channels(), etc.
    // methods as these index from the beginning of the blob shape, where legacy
    // parameter blobs were indexed from the end of the blob shape (e.g., bias
    // Blob shape (1 x 1 x 1 x N), IP layer weight Blob shape (1 x 1 x M x N)).
    return shape_.size() <= 4 &&
           LegacyShape(-4) == other.num() &&
           LegacyShape(-3) == other.channels() &&
           LegacyShape(-2) == other.height() &&
           LegacyShape(-1) == other.width();
  }
  vector<int> other_shape(other.shape().dim_size());
  for (int i = 0; i < other.shape().dim_size(); ++i) {
    other_shape[i] = other.shape().dim(i);
  }
  return shape_ == other_shape;
}

template <typename Dtype>
void Blob<Dtype>::CopyFrom(const Blob& source, bool copy_diff, bool reshape) {
  if (source.count() != count_ || source.shape() != shape_) {
    if (reshape) {
      ReshapeLike(source);
    } else {
      LOG(FATAL) << "Trying to copy blobs of different sizes.";
    }
  }
  switch (Caffe::mode()) {
  case Caffe::GPU:
    if (copy_diff) {
      caffe_copy(count_, source.gpu_diff(),
          static_cast<Dtype*>(diff_->mutable_gpu_data()));
    } else {
      caffe_copy(count_, source.gpu_data(),
          static_cast<Dtype*>(data_->mutable_gpu_data()));
    }
    break;
  case Caffe::CPU:
    if (copy_diff) {
      caffe_copy(count_, source.cpu_diff(),
          static_cast<Dtype*>(diff_->mutable_cpu_data()));
    } else {
      caffe_copy(count_, source.cpu_data(),
          static_cast<Dtype*>(data_->mutable_cpu_data()));
    }
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}

template <typename Dtype>
void Blob<Dtype>::FromProto(const BlobProto& proto, bool reshape) {
        if (reshape) {
          vector<int> shape;
          if (proto.has_num() || proto.has_channels() ||
              proto.has_height() || proto.has_width()) {
            // Using deprecated 4D Blob dimensions --
            // shape is (num, channels, height, width).
            shape.resize(4);
            shape[0] = proto.num();
            shape[1] = proto.channels();
            shape[2] = proto.height();
            shape[3] = proto.width();
          } else {
            shape.resize(proto.shape().dim_size());
            for (int i = 0; i < proto.shape().dim_size(); ++i) {
              shape[i] = proto.shape().dim(i);
            }
          }
          Reshape(shape);
        } else {
          CHECK(ShapeEquals(proto)) << "shape mismatch (reshape not set)";
        }

        // copy data
 /******************add for pruning**********************/
  if(step==2 && blob_id_==0)
  {
    if(layer_type_==1 && (layer_id_>=1 && layer_id_<=2))
    {
      //mask_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
      Dtype * mask_vec = mutable_cpu_mask();
      CHECK_EQ(count_, proto.mask_size());
      for (int i = 0; i < count_; ++i) {
        mask_vec[i] = proto.mask(i);
      }
    }
  }

  if(step==101 && blob_id_==0 && (layer_type_==1 && (layer_id_>=1 && layer_id_<=2)))
  {
      int n = proto.nnz();
      set_nnz(n);

      LOG(INFO)<<"nnz_ = "<<nnz_<<" shape_= "<<shape_[0]<<" "<<shape_[1]<<" "<<shape_[2]<<" "<<shape_[3];

      int row = shape_[0];
      csrval_.reset(new SyncedMemory(nnz_ * sizeof(Dtype)));
      csrrowptr_.reset(new SyncedMemory((row+1) * sizeof(int)));
      csrcolind_.reset(new SyncedMemory(nnz_ * sizeof(int)));
      if (proto.csrval_size() > 0) {
        CHECK_EQ(nnz_, proto.csrval_size());
        Dtype* csrval_vec = mutable_cpu_csrval();
        for (int i = 0; i < nnz_; ++i) {
          csrval_vec[i] = proto.csrval(i);
        }
      }

      if (proto.csrrowptr_size() > 0) {
        CHECK_EQ(row+1, proto.csrrowptr_size());
        int * csrrowptr_vec = mutable_cpu_csrrowptr();
        for (int i = 0; i < row+1; ++i) {
          csrrowptr_vec[i] = proto.csrrowptr(i);
        }
      }

      if (proto.csrcolind_size() > 0) {
        CHECK_EQ(nnz_, proto.csrcolind_size());
        int * csrcolind_vec = mutable_cpu_csrcolind();
        for (int i = 0; i < nnz_; ++i) {
          csrcolind_vec[i] = proto.csrcolind(i);
        }
      }
  }
  else
  {

  LOG(INFO)<<" layer_type_ =  "<<layer_type_<<" layer_id_ = "<<layer_id_;

  /*******************************************************/


        Dtype* data_vec = mutable_cpu_data();
        if (proto.double_data_size() > 0) {
          CHECK_EQ(count_, proto.double_data_size());
          for (int i = 0; i < count_; ++i) {
            data_vec[i] = proto.double_data(i);
          }
        } else if(proto.data_size() > 0){
          CHECK_EQ(count_, proto.data_size());
          for (int i = 0; i < count_; ++i) {
            data_vec[i] = proto.data(i);
          }
        }
        /// copy diff
        if (proto.double_diff_size() > 0) {
          CHECK_EQ(count_, proto.double_diff_size());
          Dtype* diff_vec = mutable_cpu_diff();
          for (int i = 0; i < count_; ++i) {
            diff_vec[i] = proto.double_diff(i);
          }
        } else if (proto.diff_size() > 0) {
          CHECK_EQ(count_, proto.diff_size());
          Dtype* diff_vec = mutable_cpu_diff();
          for (int i = 0; i < count_; ++i) {
            diff_vec[i] = proto.diff(i);
          }
        }
  /******************add for pruning**********************/
  }
  /*******************************************************/
}

template <>
void Blob<double>::ToProto(BlobProto* proto, bool write_diff) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_double_data();
  proto->clear_double_diff();
  const double* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_double_data(data_vec[i]);
  }
  if (write_diff) {
    const double* diff_vec = cpu_diff();
    for (int i = 0; i < count_; ++i) {
      proto->add_double_diff(diff_vec[i]);
    }
  }
}

template <>
void Blob<float>::ToProto(BlobProto* proto, bool write_diff) const {
  proto->clear_shape();
  for (int i = 0; i < shape_.size(); ++i) {
    proto->mutable_shape()->add_dim(shape_[i]);
  }
  proto->clear_data();
  proto->clear_diff();
 /******************add for pruning**********************/
  proto->clear_mask();
  /*generate the mask_*/
  //LOG(INFO) << "---- step = " << step;
  if(step == 1 && blob_id_==0)
  {
    if(layer_type_==1 && (layer_id_>=1 && layer_id_<=2))
    {
      LOG(INFO) << "save the mask ----";
      const float* data_vec_copy = cpu_data();

      float max = -1000.0;
      float min = 1000.0;
      int count_0 = 0;
      int count_1 = 0;

      LOG(INFO)<<"count_ = "<<count_;

      for (int i = 0; i < count_; ++i) {

        //LOG(INFO)<<data_vec_copy[i];
        if(max<data_vec_copy[i])
        {
          max = data_vec_copy[i];
        }
        if(min>data_vec_copy[i])
        {
          min = data_vec_copy[i];
        }
        float thre;
        if(layer_id_ == 1)
        {
           thre = thre_fc6;
        }
        else
        {
           thre = thre_fc7;
        }

        int mask_v;
        if(data_vec_copy[i]>(0.0-thre) && data_vec_copy[i]<thre)
        {
          mask_v = 0;
          count_0++;
        }
        else
        {
          mask_v = 1;
          count_1++;
        }
        //LOG(INFO)<<mask_v;

        proto->add_mask(mask_v);
      }

      LOG(INFO)<<"max = "<<max;
      LOG(INFO)<<"min = "<<min;

      float total = count_;
      float total_0 = count_0;
      float total_1 = count_1;
      LOG(INFO)<<"count_0 = "<<count_0<<"  "<<total_0/total;
      LOG(INFO)<<"count_1 = "<<count_1<<"  "<<total_1/total;
    }
  }

  if(step == 2 && blob_id_==0 && (layer_type_==1 && (layer_id_>=1 && layer_id_<=2)))
  {
      cusparseHandle_t cphandle;
      cusparseCreate(&cphandle);

      cusparseMatDescr_t descr;
      cusparseCreateMatDescr(&descr);
      cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
      cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

      int row = shape_[0];
      int col = shape_[1];
      /*
      LOG(INFO) <<"row = "<< row <<" col = "<< col;
      */
      int * dev_dNnzPerRow;
      cudaMalloc((void **)&dev_dNnzPerRow,sizeof(int)*row);

      float * dev_weight;
      cudaMalloc((void **)&dev_weight,sizeof(float)*row*col);
      cublasHandle_t cbhandle;
      cublasCreate_v2(&cbhandle);
      float alpha = 1.0;
      float beta = 0.0;
      cublasSgeam(cbhandle,CUBLAS_OP_T,CUBLAS_OP_N,row,col,&alpha,gpu_data(),col,&beta,gpu_data(),row,dev_weight,row);
      cudaDeviceSynchronize();

      int totalNnz;
      cusparseSnnz(cphandle,CUSPARSE_DIRECTION_ROW,row,col,descr,dev_weight,row,dev_dNnzPerRow,&totalNnz);
      cudaDeviceSynchronize();

      LOG(INFO) <<"save the csr..........totalNnz = "<<totalNnz;

      float * dev_csrval;
      cudaMalloc((void **)&dev_csrval,totalNnz * sizeof(float));
      int * dev_csrrowptr;
      cudaMalloc((void **)&dev_csrrowptr,(row + 1) * sizeof(int));
      int * dev_csrcolind;
      cudaMalloc((void **)&dev_csrcolind,totalNnz * sizeof(int));

      cusparseSdense2csr(cphandle,row,col,descr,dev_weight,row,dev_dNnzPerRow,dev_csrval,dev_csrrowptr,dev_csrcolind);
      cudaDeviceSynchronize();

      float * host_csrval;
      host_csrval = (float *)malloc(totalNnz * sizeof(float));
      cudaMemcpy(host_csrval,dev_csrval,totalNnz * sizeof(float),cudaMemcpyDeviceToHost);
      int * host_csrrowptr;
      host_csrrowptr = (int *)malloc((row + 1) * sizeof(int));
      cudaMemcpy(host_csrrowptr,dev_csrrowptr,(row + 1) * sizeof(int),cudaMemcpyDeviceToHost);
      int * host_csrcolind;
      host_csrcolind = (int *)malloc(totalNnz * sizeof(int));
      cudaMemcpy(host_csrcolind,dev_csrcolind,totalNnz * sizeof(int),cudaMemcpyDeviceToHost);
      /*
      LOG(INFO)<<"csrval_ = "<<host_csrval[0]<<" "<<host_csrval[1]<<" "<<host_csrval[totalNnz-2]<<" "<<host_csrval[totalNnz-1];
      LOG(INFO)<<"csrrowptr_ = "<<host_csrrowptr[0]<<" "<<host_csrrowptr[1]<<" "<<host_csrrowptr[row-1]<<" "<<host_csrrowptr[row];
      LOG(INFO)<<"csrcolind_ = "<<host_csrcolind[0]<<" "<<host_csrcolind[1]<<" "<<host_csrcolind[totalNnz-2]<<" "<<host_csrcolind[totalNnz-1];
      */
      proto->set_nnz(totalNnz);
      for(int i = 0;i < totalNnz; ++i){
        proto->add_csrval(host_csrval[i]);
      }
      for(int i = 0;i < row+1; ++i){
        proto->add_csrrowptr(host_csrrowptr[i]);
      }
      for(int i = 0;i < totalNnz; ++i){
        proto->add_csrcolind(host_csrcolind[i]);
      }

      free(host_csrval);
      free(host_csrrowptr);
      free(host_csrcolind);
      cudaFree(dev_weight);
      cudaFree(dev_dNnzPerRow);
      cudaFree(dev_csrval);
      cudaFree(dev_csrrowptr);
      cudaFree(dev_csrcolind);
      cublasDestroy_v2(cbhandle);
      cusparseDestroy(cphandle);
  }
  else
  {
    const float* data_vec = cpu_data();
    for (int i = 0; i < count_; ++i) {
      proto->add_data(data_vec[i]);
    }
    if (write_diff) {
      const float* diff_vec = cpu_diff();
      for (int i = 0; i < count_; ++i) {
        proto->add_diff(diff_vec[i]);
      }
    }
  }
  /********************************************************/
  /*
  const float* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
      proto->add_data(data_vec[i]);
  }
  if (write_diff) {
      const float* diff_vec = cpu_diff();
      for (int i = 0; i < count_; ++i) {
          proto->add_diff(diff_vec[i]);
      }
  }
  */
}

template <typename Dtype>
void Blob<Dtype>::WriteTxtTo(std::string filenm){
    FILE* fp = NULL;
    if((fp = fopen(filenm.c_str(), "wt")) == NULL)
        LOG(ERROR) << "Can't open file [" << filenm << "]!";

    std::stringstream output;
    output << "Blob shape:" << std::endl << shape_string() << std::endl;
    for(int i = 0; i < count(); i++){
        output << cpu_data()[i] << " ";
        if(i % 20 == 0)
            output << std::endl;
    }
    fprintf(fp, "%s\n", output.str().c_str());
    fclose(fp);
}

INSTANTIATE_CLASS(Blob);
template class Blob<int>;
template class Blob<unsigned int>;

}  // namespace caffe


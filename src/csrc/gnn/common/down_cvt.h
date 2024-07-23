
auto out_ptr = GetVLAPtr<T>(out_t, {feat_dim});
auto cvt_in_out = ConvertTPP<float, T>(1, feat_dim);

#pragma omp parallel
{
  int nthreads = omp_get_num_threads();
  int tid = omp_get_thread_num();
  long work = rows % nthreads == 0 ? rows / nthreads : rows / nthreads + 1;
  at::Tensor in_t = at::empty({feat_dim});
  auto in_ptr = in_t.data_ptr<float>();

  long tb = tid * work < rows ? tid * work : rows;
  long te = (tid + 1) * work < rows ? (tid + 1) * work : rows;
  FILE* f = fopen(in_fn.c_str(), "rb");
  fseek(f, tb * feat_dim * sizeof(float), SEEK_SET);
  for (long i = tb; i < te; i++) {
    fread((void*)in_ptr, sizeof(float), feat_dim, f);
    cvt_in_out(in_ptr, out_ptr[i]);
  }
  fclose(f);
}

return out_t;

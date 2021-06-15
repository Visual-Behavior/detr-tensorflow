# With tf env activated
TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

nvcc -std=c++11 -c -o ms_deform_im2col.o ms_deform_im2col.cu.cc ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr

g++ -std=c++11 -shared -o ms_deform_im2col.so ms_deform_im2col.cc ms_deform_im2col.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]}

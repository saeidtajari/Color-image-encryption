N=6500000
r(){ name="$1"; shift; ( time "$@" > out_$name.bin ) 2> t_$name.txt; echo "$name $(sha256sum < out_$name.bin | cut -c1-16) $(grep real t_$name.txt)"; }
r c_x86 ./c_x86 $N /dev/stdout
r c_clang ./c_clang $N /dev/stdout
r go_x86 ./go_x86 $N
r go_x86_naive ./go_x86_naive $N
r rs_x86 ./rs_x86 $N
r java java -cp . M $N 2>/dev/null
r node node m.js $N
r c_a64_off qemu-aarch64-static ./c_a64_off $N /dev/stdout
r c_a64_def qemu-aarch64-static ./c_a64_def $N /dev/stdout
r c_arm32 qemu-arm-static ./c_arm32 $N /dev/stdout
r go_a64_guard qemu-aarch64-static ./go_a64_guard $N
r go_a64_naive qemu-aarch64-static ./go_a64_naive $N

#!/bin/bash
# Builds and runs the map + output filter (Eq. 10) in several languages and toolchains; prints SHA-256 of the q values.
set -e
N=${1:-8500000}
gcc -O2 -ffp-contract=off q.c -o c_x86 -lm
gcc -O3 -march=native -ffp-contract=fast q.c -o c_x86_fma -lm
clang -O2 q.c -o c_clang -lm
gcc -O3 -ffast-math q.c -o c_fastmath -lm
aarch64-linux-gnu-gcc -O2 -ffp-contract=off -static q.c -o c_a64_off -lm
aarch64-linux-gnu-gcc -O2 -static q.c -o c_a64_def -lm
arm-linux-gnueabihf-gcc -O2 -ffp-contract=off -static q.c -o c_arm32 -lm
go build -o go_x86_guard q_true.go; go build -o go_x86_naive q_false.go
GOARCH=arm64 go build -o go_a64_guard q_true.go; GOARCH=arm64 go build -o go_a64_naive q_false.go
rustc -O q.rs -o rs_x86 2>/dev/null
javac Q.java
r(){ name="$1"; shift; "$@" > out_$name.bin; echo "$name $(sha256sum < out_$name.bin | cut -c1-16)"; }
r c_x86 ./c_x86 $N /dev/stdout
r c_x86_fma ./c_x86_fma $N /dev/stdout
r c_clang ./c_clang $N /dev/stdout
r c_fastmath ./c_fastmath $N /dev/stdout
r go_x86_guard ./go_x86_guard $N
r go_x86_naive ./go_x86_naive $N
r rs_x86 ./rs_x86 $N
r java java -cp . Q $N
r node node q.js $N
r c_a64_off qemu-aarch64-static ./c_a64_off $N /dev/stdout
r c_a64_def qemu-aarch64-static ./c_a64_def $N /dev/stdout
r c_arm32 qemu-arm-static ./c_arm32 $N /dev/stdout
r go_a64_guard qemu-aarch64-static ./go_a64_guard $N
r go_a64_naive qemu-aarch64-static ./go_a64_naive $N

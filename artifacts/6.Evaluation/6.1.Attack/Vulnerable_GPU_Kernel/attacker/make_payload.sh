nvcc -G -arch=sm_89 payload.cu
cuobjdump --dump-sass a.out > payload.sass

python3 parser.py

mv input.bin ../

rm a.out payload.sass

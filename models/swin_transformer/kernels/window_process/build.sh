srun -p V100 -K \
    -N 1 \
    -w $1 \
    --export=ALL \
    --ntasks-per-node=1 \
    --gres=gpu:1 \
    python setup.py install

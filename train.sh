# default
python -m torch.distributed.launch \
	--nproc_per_node=4 \
	--use_env main.py \
	--model deit_tiny_patch16_224 \
	--batch-size 256 \
	--data-path /path/to/imagenet \
	--output_dir outputs/deit_tiny_patch16_224

# without pos
python -m torch.distributed.launch \
	--nproc_per_node=4 \
	--use_env main.py \
	--model deit_tiny_patch16_224_without_pos \
	--batch-size 256 \
	--data-path /path/to/imagenet \
	--output_dir outputs/deit_tiny_patch16_224_without_pos

# with sinusoidal pos
python -m torch.distributed.launch \
	--nproc_per_node=4 \
	--use_env main.py \
	--model deit_tiny_patch16_224_with_sin \
	--batch-size 256 \
	--data-path /path/to/imagenet \
	--output_dir outputs/deit_tiny_patch16_224_with_sin

# with aaud
python -m torch.distributed.launch \
	--nproc_per_node=4 \
	--use_env main.py \
	--model deit_tiny_patch16_224_with_aaud \
	--batch-size 256 \
	--data-path /path/to/imagenet \
	--output_dir outputs/deit_tiny_patch16_224_with_aaud

# with naive ae
python -m torch.distributed.launch \
	--nproc_per_node=4 \
	--use_env main.py \
	--model deit_tiny_patch16_224_with_naive_ae \
	--batch-size 256 \
	--data-path /path/to/imagenet \
	--output_dir outputs/deit_tiny_patch16_224_with_naive_ae


python -m torch.distributed.launch \
	--nproc_per_node=3 \
	--use_env main.py \
	--model deit_tiny_patch16_224_with_naive_ae \
	--batch-size 192 \
	--data-path /workspace/dataset/ILSVRC2012 \
	--output_dir outputs/deit_tiny_patch16_224_with_naive_ae
# InstructPix2Pix
UnOfficial Pytorch implementation of 'InstructPix2Pix Learning to Follow Image Editing Instructions', based on https://github.com/JoePenna/Dreambooth-Stable-Diffusion

# Inference
python3 stable_txt2img.py --ddim_eta 0.0 --n_samples 4 --n_iter 1 --ddim_steps 50 --ckpt logs/instruct/checkpoints/last.ckpt --W 256 --H 256 --init_img ./samples/tower.jpg --prompt "add fireworks in sky"
![result](https://github.com/xuduo35/InstructPix2Pix/raw/main/samples/add-fireworks-in-sky-0000.jpg )

python3 stable_txt2img.py --ddim_eta 0.0 --n_samples 4 --n_iter 1 --ddim_steps 50 --ckpt logs/instruct/checkpoints/last.ckpt --W 256 --H 256 --init_img ./samples/tower.jpg --prompt "add fireworks in sky" --negprompt "blurred"
![result](https://github.com/xuduo35/InstructPix2Pix/raw/main/samples/add-fireworks-in-sky-0005-blurred.jpg )

python3 stable_txt2img.py --ddim_eta 0.0 --n_samples 4 --n_iter 1 --ddim_steps 50 --ckpt logs/instruct/checkpoints/last.ckpt --W 512 --H 512 --init_img ./samples/Vermeer_Girl.jpg --prompt "Apply face paint"
![result](https://github.com/xuduo35/InstructPix2Pix/raw/main/samples/Apply-face-paint-0000-512x512.jpg )

python3 stable_txt2img.py --ddim_eta 0.0 --n_samples 4 --n_iter 1 --ddim_steps 50 --ckpt logs/instruct/checkpoints/last.ckpt --W 512 --H 512 --init_img ./training_images/Vermeer_Girl.jpg --prompt "What if she were in an anime?"
![result](https://github.com/xuduo35/InstructPix2Pix/raw/main/samples/What-if-she-were-in-an-anime%3F-0000-512x512.jpg )

python3 stable_txt2img.py --ddim_eta 0.0 --n_samples 4 --n_iter 1 --ddim_steps 50 --ckpt logs/instruct/checkpoints/last.ckpt --W 512 --H 512 --init_img ./samples/Vermeer_Girl.jpg --prompt "Put on a pair of sunglasses?"
![result](https://github.com/xuduo35/InstructPix2Pix/raw/main/samples/Put-on-a-pair-of-sunglasses-0005-512x512.jpg )

python3 stable_txt2img.py --ddim_eta 0.0 --n_samples 4 --n_iter 1 --ddim_steps 50 --ckpt logs/instruct/checkpoints/last.ckpt --W 256 --H 256 --init_img ./training_images/dog.jpg --prompt "pig"
![result](https://github.com/xuduo35/InstructPix2Pix/raw/main/samples/pig-0000.jpg )

python3 stable_txt2img.py --ddim_eta 0.0 --n_samples 4 --n_iter 1 --ddim_steps 50 --ckpt logs/instruct/checkpoints/last.ckpt --W 256 --H 256 --init_img ./samples/dog.jpg --prompt "dog in Paris"
![result](https://github.com/xuduo35/InstructPix2Pix/raw/main/samples/dog-in-Paris-0000.jpg )

python3 stable_txt2img.py --ddim_eta 0.0 --n_samples 4 --n_iter 1 --ddim_steps 50 --ckpt logs/instruct/checkpoints/last.ckpt --W 256 --H 256 --init_img ./samples/sunflowers.jpg --prompt "roses"
![result](https://github.com/xuduo35/InstructPix2Pix/raw/main/samples/roses-0000.jpg )

python3 stable_txt2img.py --ddim_eta 0.0 --n_samples 4 --n_iter 1 --ddim_steps 50 --ckpt logs/instruct/checkpoints/last.ckpt --W 256 --H 256 --init_img ./samples/girl.jpg --prompt "She should look 100 years old" --negprompt "deformed"
![result](https://github.com/xuduo35/InstructPix2Pix/raw/main/samples/She-should-look-100-years-old-0000-deformed.jpg )

python3 stable_txt2img.py --ddim_eta 0.0 --n_samples 4 --n_iter 1 --ddim_steps 50 --ckpt logs/instruct/checkpoints/last.ckpt --W 512 --H 512 --init_img ./samples/girl.jpg --prompt "make hair red"
![result](https://github.com/xuduo35/InstructPix2Pix/raw/main/samples/make-hair-red-0000-512x512.jpg )

python3 stable_txt2img.py --ddim_eta 0.0 --n_samples 4 --n_iter 1 --ddim_steps 50 --ckpt logs/instruct/checkpoints/last.ckpt --W 512 --H 512 --init_img ./samples/girl.jpg --prompt "make hair curly"
![result](https://github.com/xuduo35/InstructPix2Pix/raw/main/samples/make-hair-curly-0000-512x512.jpg )

# Checkpoint
Link: https://drive.google.com/file/d/1vn9qG4kLvXPNJAT-PW7Exwas7MyT7JBu/view?usp=sharing

# Implementation deatils
1. Add additional input channels to the first convolutional layer. All available weights of the diffusion model are initialized from the pretrained checkpoints, and weights that operate on the newly added input channels are initialized to zero. Besides, I add one more GroupNorm32/SiLU/conv_nd layer than original paper.
2. Set learing rate set 1e-4, batch size 32

# Data Preperation
This is tough and money-consuming part...

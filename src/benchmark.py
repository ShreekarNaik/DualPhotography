import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform, color, img_as_float
import json
import os
import glob
from datetime import datetime

from transforms import WaveletTransform, DCTTransform, HadamardTransform
from strategy import RandomStrategy, LowFreqStrategy, AdaptiveOracleStrategy
from experiment import Experiment

def load_images_from_data(size=(256, 256)):
    data_dir = "data"
    image_files = glob.glob(os.path.join(data_dir, "*"))
    images = []
    names = []
    
    for filepath in image_files:
        try:
            # Load image
            img = io.imread(filepath)
            
            # Convert to grayscale if RGB
            if img.ndim == 3:
                img = color.rgb2gray(img)
            
            # Resize
            img = transform.resize(img, size)
            
            # Ensure float 0-1
            img = img_as_float(img)
            
            images.append(img)
            names.append(os.path.basename(filepath))
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            
    return images, names

def run_benchmark():
    # Setup Output Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/image/run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to {output_dir}")
    
    # Load Images
    images, names = load_images_from_data(size=(256, 256))
    if not images:
        print("No images found in data/. Using default camera image.")
        from skimage import data
        img = data.camera()
        img = transform.resize(img, (256, 256))
        img = (img - img.min()) / (img.max() - img.min())
        images = [img]
        names = ["camera.png"]
    
    # Run Benchmark for each image
    for img, img_name in zip(images, names):
        print(f"\nProcessing {img_name}...")
        
        # Save Original
        io.imsave(os.path.join(output_dir, f"{img_name}_original.png"), (img * 255).astype(np.uint8))
        
        max_measurements = int(0.05 * img.size) # 5% sampling
        
        # Define Configurations
        configs = []
        
        # 1. Wavelet
        wt = WaveletTransform('db1', levels=3)
        wt_true = wt.forward(img)
        configs.append((wt, RandomStrategy(len(wt_true), wt.shape), "Wavelet_Random"))
        configs.append((wt, AdaptiveOracleStrategy(len(wt_true), wt.shape, wt_true), "Wavelet_Oracle"))
        
        # 2. DCT
        dct = DCTTransform()
        dct_true = dct.forward(img)
        configs.append((dct, RandomStrategy(len(dct_true), img.shape), "DCT_Random"))
        configs.append((dct, LowFreqStrategy(len(dct_true), img.shape), "DCT_LowFreq"))
        
        # 3. Hadamard
        ht = HadamardTransform()
        ht_true = ht.forward(img)
        configs.append((ht, RandomStrategy(len(ht_true), img.shape), "Hadamard_Random"))
        configs.append((ht, LowFreqStrategy(len(ht_true), img.shape), "Hadamard_LowFreq"))
        
        # Run Experiments
        results_data = {}
        
        plt.figure(figsize=(10, 6))
        
        for trans, strat, name in configs:
            # Create directory for this specific run
            run_dir = os.path.join(output_dir, img_name, name)
            os.makedirs(run_dir, exist_ok=True)
            
            patterns_dir = os.path.join(run_dir, "patterns")
            
            exp = Experiment(img, trans, strat, max_measurements=max_measurements, log_interval=200, save_patterns_dir=patterns_dir)
            res, rec_img = exp.run()
            results_data[name] = res
            
            # Save Reconstructed Image
            save_name = f"reconstructed.png"
            io.imsave(os.path.join(run_dir, save_name), (rec_img * 255).astype(np.uint8))
            
            # Plot (Individual)
            plt.plot(res["measurements"], res["psnr"], label=name, linewidth=2)
            
        # Save Combined Plot
        plt.xlabel("Measurements")
        plt.ylabel("PSNR (dB)")
        plt.title(f"Adaptive CS Benchmark: {img_name}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{img_name}_plot.png"))
        plt.close()
        
        # Save Data
        with open(os.path.join(output_dir, f"{img_name}_data.json"), "w") as f:
            json.dump(results_data, f, indent=4)

    print(f"\nBenchmark Complete. All results saved to {output_dir}")

if __name__ == "__main__":
    run_benchmark()

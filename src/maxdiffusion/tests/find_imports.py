import inspect
from diffusers import Flux2KleinPipeline

def main():
    print("Finding VAE and Scheduler imports in Flux2KleinPipeline...")
    file_path = inspect.getfile(Flux2KleinPipeline)
    print(f"Pipeline file path: {file_path}")
    
    print("\n=======================================================")
    print("First 80 lines of the pipeline file (Imports):")
    print("=======================================================")
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
            for i in range(min(80, len(lines))):
                print(f"{i+1:3d}: {lines[i]}", end="")
    except Exception as e:
        print(f"Failed to read file: {e}")

if __name__ == "__main__":
    main()

# Run all demos for pirate.tif and synthetic images, for all operators, backends, and detection methods. 
python demo.py --image pirate.tif --operator prewitt --backend statevector --detection response --skip-existing
python demo.py --image pirate.tif --operator prewitt --backend statevector --detection binary --skip-existing
python demo.py --image pirate.tif --operator prewitt --backend noisy --detection response --skip-existing 
python demo.py --image pirate.tif --operator prewitt --backend noisy --detection binary --skip-existing

python demo.py --image pirate.tif --operator sobel --backend statevector --detection response --skip-existing 
python demo.py --image pirate.tif --operator sobel --backend statevector --detection binary --skip-existing
python demo.py --image pirate.tif --operator sobel --backend noisy --detection response --skip-existing 
python demo.py --image pirate.tif --operator sobel --backend noisy --detection binary --skip-existing

python demo.py --image pirate.tif --operator kirsch --backend statevector --detection response --skip-existing 
python demo.py --image pirate.tif --operator kirsch --backend statevector --detection binary --skip-existing
python demo.py --image pirate.tif --operator kirsch --backend noisy --detection response --skip-existing 
python demo.py --image pirate.tif --operator kirsch --backend noisy --detection binary --skip-existing

python demo.py --image synthetic --operator prewitt --backend statevector --detection response --skip-existing 
python demo.py --image synthetic --operator prewitt --backend statevector --detection binary --skip-existing
python demo.py --image synthetic --operator prewitt --backend noisy --detection response --skip-existing 
python demo.py --image synthetic --operator prewitt --backend noisy --detection binary --skip-existing

python demo.py --image synthetic --operator sobel --backend statevector --detection response --skip-existing 
python demo.py --image synthetic --operator sobel --backend statevector --detection binary --skip-existing
python demo.py --image synthetic --operator sobel --backend noisy --detection response --skip-existing 
python demo.py --image synthetic --operator sobel --backend noisy --detection binary --skip-existing

python demo.py --image synthetic --operator kirsch --backend statevector --detection response --skip-existing 
python demo.py --image synthetic --operator kirsch --backend statevector --detection binary --skip-existing
python demo.py --image synthetic --operator kirsch --backend noisy --detection response --skip-existing 
python demo.py --image synthetic --operator kirsch --backend noisy --detection binary --skip-existing

# Generate plots for the results of the pirate and synthetic images with kirsch operator, noisy backend, binary detection method, and magnitude normalization. You can generate plots for other combinations by changing the parameters accordingly.
python plot_dual_axis.py results/pirate_kirsch_noisy_binary_otsu_magnorm/qced_results_pirate_noisy.xlsx  --no-classical
python plot_dual_axis.py results/synthetic_kirsch_noisy_binary_otsu_magnorm/qced_results_synthetic_noisy.xlsx  --no-classical

# Run demos for synthetic images with QPU backend for both response and binary detection methods.
# You have to fill ibm_account.py with your IBM Quantum account credentials to run the QPU backend demos.
# Once you run the QPU with 'run_qpu_experiment.py', you can reuse the results for the other detection method with the 'demo.py' script.
python ibm_account.py
python run_qpu_experiment.py --image synthetic.tif --operator kirsch --shots 1024 --chunk-size 1 --simulate # --simulate flag is used to simulate the QPU execution without actually running it.

python run_qpu_experiment.py --image synthetic.tif --operator kirsch --shots 512 --chunk-size 1 --backend-name ibm_fez --run-id 1
python demo.py --image synthetic --operator kirsch --backend qpu --detection response --sweep 512 --seeds 1 --seed 0 --qpu-run-id 1
python demo.py --image synthetic --operator kirsch --backend qpu --detection binary   --sweep 512 --seeds 1 --seed 0 --qpu-run-id 1

python run_qpu_experiment.py --image synthetic.tif --operator kirsch --shots 512 --chunk-size 1 --backend-name ibm_fez --run-id 2
python demo.py --image synthetic --operator kirsch --backend qpu --detection response --sweep 512 --seeds 1 --seed 1 --qpu-run-id 2
python demo.py --image synthetic --operator kirsch --backend qpu --detection binary   --sweep 512 --seeds 1 --seed 1 --qpu-run-id 2

python run_qpu_experiment.py --image synthetic.tif --operator kirsch --shots 512 --chunk-size 1 --backend-name ibm_fez --run-id 3
python demo.py --image synthetic --operator kirsch --backend qpu --detection response --sweep 512 --seeds 1 --seed 2 --qpu-run-id 3
python demo.py --image synthetic --operator kirsch --backend qpu --detection binary   --sweep 512 --seeds 1 --seed 2 --qpu-run-id 3

python run_qpu_experiment.py --image synthetic.tif --operator kirsch --shots 1024 --chunk-size 1 --backend-name ibm_fez --run-id 1
python demo.py --image synthetic --operator kirsch --backend qpu --detection response --sweep 1024 --seeds 1 --seed 0 --qpu-run-id 1
python demo.py --image synthetic --operator kirsch --backend qpu --detection binary   --sweep 1024 --seeds 1 --seed 0 --qpu-run-id 1

python run_qpu_experiment.py --image synthetic.tif --operator kirsch --shots 1024 --chunk-size 1 --backend-name ibm_fez --run-id 2
python demo.py --image synthetic --operator kirsch --backend qpu --detection response --sweep 1024 --seeds 1 --seed 1 --qpu-run-id 2
python demo.py --image synthetic --operator kirsch --backend qpu --detection binary   --sweep 1024 --seeds 1 --seed 1 --qpu-run-id 2

python run_qpu_experiment.py --image synthetic.tif --operator kirsch --shots 1024 --chunk-size 1 --backend-name ibm_fez --run-id 3
python demo.py --image synthetic --operator kirsch --backend qpu --detection response --sweep 1024 --seeds 1 --seed 2 --qpu-run-id 3
python demo.py --image synthetic --operator kirsch --backend qpu --detection binary   --sweep 1024 --seeds 1 --seed 2 --qpu-run-id 3

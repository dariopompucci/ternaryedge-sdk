# ternaryedge-sdk
 TernaryEdge SDK – A PyTorch toolkit for training and deploying ultra-efficient ternary neural networks (weights in {-1, 0, +1}) on resource-constrained edge devices. Inc. quantization flows (TWN/TTQ), export to packed ternary formats, early support for NEON/AVX2 inference. Built for keyword spotting, anomaly detection, MLPerf Tiny-class tasks.
#  TernaryEdge SDK

*Efficient neural networks with {-1, 0, +1} weights for the edge. Because floating-point is for cowards.*

//-----------------------------------------------------------------------------------------------------------//

##  What is this?

**TernaryEdge SDK** is a framework for training and deploying **ternary neural networks** — models where weights and activations are constrained to three possible values: **-1, 0, +1**.

Why ternary? Because:
- You get **~16× smaller models** without severe accuracy loss.
- You can skip actual multiplies — **add, subtract, or skip**.
- You can map them efficiently to **low-power CPUs, FPGAs, and even exotic 3-level memory hardware** (RRAM, PCM, etc).

We're building the full stack — from **PyTorch training** to **on-device inference kernels**, with hardware-aware quantization and ultra-low power deployment in mind.

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------//

##  Who is this for?

- TinyML engineers tired of bloated models
- Embedded systems devs working with <1W budgets
- Researchers exploring **balanced ternary** and **MVL memory**
- Masochists with dreams of building efficient AI on toasters

//---------------------------------------------------------------//

##  Repo Structure (WIP)

ternaryedge-sdk/
├── models/ # PyTorch model definitions
├── quant/ # Ternary quantization methods (TWN, TTQ, etc)
├── training/ # Training scripts + config
├── export/ # Tools to convert to inference formats (int2)
├── backends/ # NEON/AVX2/C-based inference kernels
├── tests/ # Unit tests and sanity checks
├── notebooks/ # Jupyter demos + benchmarks
└── README.md # You are here



//--------------------------------------------------------//

## Features

- ✅ PyTorch modules with TWN / TTQ ternary quantization
- ✅ Straight-through estimator (STE) for ternary layers
- 🔜 2-bit packed model export for deployment
- 🔜 AVX2 / NEON optimized inference backends
- 🔜 FPGA prototype (TriMAC core)
- 🔜 ONNX export + CLI tools

//---------------------------------------------------------//
Roadmap (updated live-ish)

 Train TWN/TTQ KWS model on Google Speech Commands

 2-bit weight export + model converter

 CPU inference kernels (NEON, AVX2)

 MLPerf Tiny benchmark submission

 TriMAC FPGA prototype (Lattice iCE40?)

 Sky130 ASIC tapeout (eventually)

//---------------------------------------------------------//
##  Quickstart (once it exists)

```bash
# Clone the repo
git clone https://github.com/<yourname>/ternaryedge-sdk.git
cd ternaryedge-sdk

# Install dependencies
pip install -r requirements.txt

# Train a ternary model (e.g. keyword spotting)
python train.py --model ternary_kws --dataset speech_commands

# Export model
python export/export_model.py --checkpoint path/to/model.ckpt
//--------------------------------------------------------------//


📊 Benchmarks

We're targeting MLPerf Tiny workloads:

Keyword Spotting (Google Speech Commands)

Visual Wake Word (person presence)

Anomaly Detection (vibration, industrial)

Initial benchmarks coming soon. Expect:

✅ ~16x smaller model size vs FP32

✅ <10% accuracy drop

✅ >4x inference speedup on CPU (vs PyTorch baseline)

✅ Laughably low energy use on ARM and FPGA

 Why Ternary?

3 values = smaller weights

0s = skip MACs entirely

-1/+1 = add/sub only (no multiply)

Easy to compress, faster to compute

Maps well to post-CMOS memory (RRAM, PCM, FeFET)

Inspired by:

TWN, TTQ, BinaryConnect

Setun (1959 ternary computer)

A growing hatred for quantization errors
//--------------------------------------------------//
 Commercial Use

We’re exploring:

Design services (ternary model tuning + FPGA IP)

Licensing of SDK or accelerator IP (TriMAC)

Custom deployments (tiny keyword spotting, anomaly detection, VWW)

Interested?
Reach out: (I didn't make an email specifically for the project yet...)
//-----------------------------------------------------------------------------//
 License

Apache 2.0 — Open for research & tinkering.
Contact for licensing if you're building commercial hardware/products.

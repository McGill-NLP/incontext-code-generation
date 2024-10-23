<h2 align="center">
  Evaluating In-Context Learning of Libraries for Code Generation
</h2>
<!-- <h5 align="center">Evaluating the In-Context Learning Ability of Large Language Models to Generalize to Novel Interpretations</h5> -->

<p align="center">
  <a href="https://2024.naacl.org/"><img src="https://img.shields.io/badge/NAACL-2024-blue"></a>
  <a href="https://aclanthology.org/2024.naacl-long.161.pdf"><img src="http://img.shields.io/badge/Paper-PDF-red.svg"></a>
  <a href="https://github.com/McGill-NLP/incontext-code-generation/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green">
  </a>
</p>

<p style="text-align: justify;">
Contemporary Large Language Models (LLMs) exhibit a high degree of code generation and comprehension capability. A particularly promising area is their ability to interpret code modules from unfamiliar libraries for solving user-instructed tasks. Recent work has shown that large proprietary LLMs can learn novel library usage in-context from demonstrations. These results raise several open questions: whether demonstrations of library usage is required, whether smaller (and more open) models also possess such capabilities, etc. In this work, we take a broader approach by systematically evaluating a diverse array of LLMs across three scenarios reflecting varying levels of domain specialization to understand their abilities and limitations in generating code based on libraries defined in-context. Our results show that even smaller open-source LLMs like Llama-2 and StarCoder demonstrate an adept understanding of novel code libraries based on specification presented in-context. Our findings further reveal that LLMs exhibit a surprisingly high proficiency in learning novel library modules even when provided with just natural language descriptions or raw code implementations of the functions, which are often cheaper to obtain than demonstrations. Overall, our results pave the way for harnessing LLMs in more adaptable and dynamic coding environments.
</p>
<h2 align="center">
  <img align="center"  src="./images/Fig_main.svg" alt="..." width="800">
</h2>



# Setup

Install VirtualEnv using the following (optional):

```shell
$ [sudo] pip install virtualenv
```

Create and activate your virtual environment (optional):

```shell
$ virtualenv -p python3 venv
$ source venv/bin/activate
```

## Dependencies

- compatible with python 3
- dependencies can be installed using `incontext-code-generation/requirements.txt`

Install all the required packages:

at `incontext-code-generation/:`

```shell
$ pip install -r requirements.txt
```

# Usage

Here, we illustrate running the experiments for GPT-4o-mini and Llama3.1 for specific experimental settings across the three domains. Follow the same methodology for running any experiment over any model.

## I. Learning a Novel Library - VisProg

### Download images

The images for Knowledge Tagging and Image Editing are not publicly available.

The images for GQA can be downloaded from [this](https://cs.stanford.edu/people/dorarad/gqa/about.html) webpage. Store images in `visprog/visprog_code/data/gqa/images/`.

The images for NLVR can be downloaded from [here](https://drive.google.com/file/d/1aftPiK2NRKUfQNEjVFt9_NyO3ZQmHry5/view?usp=sharing). If you use NLVR in your work, we request you to cite the [original paper](https://arxiv.org/abs/1811.00491). Store images in `visprog/visprog_code/data/nlvr/nlvr2/images/`.

### System-dependent changes to code

In `visprog/visprog_code/engine/utils.py`, L54, enter your openai API key.

### Generating Programs

The code for generating programs is in `visprog/generate.py`.
For example, to call GPT-4o-mini, with the demonstrations prompt for the GQA dataset,

At `visprog/`:
```shell
$ python generate.py -data gqa -prompt_type demos -model_type chat -model gpt-4o-mini -temperature 0.5
```

This will create a new directory within `visprog/results` that looks like `gqa_demos_*` which will store the model generated proofs in a `test_generated_programs.csv` file.

As another example, lets generate programs for NLVR using the descriptions prompt using Llama3.1-70B:

Host the Llama model locally using [vllm](https://github.com/vllm-project/vllm):

```shell
$ CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3.1-70B-Instruct --download-dir YOUR_DOWNLOAD_DIR --tensor-parallel-size 2 --rope-scaling='{"type": "extended", "factor": 8.0}' --max_model_len 4000 --gpu_memory_utilization 0.95
```

Then at `visprog/`:
```shell
$ python generate.py -data nlvr -prompt_type docstring_all -model_type vllm -model meta-llama/Meta-Llama-3.1-70B-Instruct
```

### Evaluating Programs

The code for generating programs is in `visprog/evaluate.py`.
For example, to evaluate the above gpt-4o-mini-generated programs for GQA,

At `visprog/`:

```shell
$ python evaluate.py -data gqa -progs_name gqa_demos_*
```

## III. Learning a Novel Programming Language - Isabelle

### Install PISA and Isabelle

Follow the instructions [here](https://github.com/wellecks/ntptutorial/blob/main/partII_dsp/isabelle_setup.md) to setup PISA and Isabelle which we will need to automatically evaluate generated proofs (follow till and including "Setup a working directory and working file").

Note:
- Use [this link](https://isabelle.in.tum.de/website-Isabelle2022/dist/Isabelle2022_linux.tar.gz) to download Isabelle using wget instead of the one provided in the above instructions.
- If you are using a more recent version of Java, use `sbt -Djava.security.manager=allow "runMain pisa.server.PisaOneStageServer9000"` instead.

### System-dependent changes to code

In `isabelle/generate.py`, L158, enter your openai API key.

In `isabelle/evaluate.py`, L7-8, change the paths for Isabelle and PISA, depending on where in your system you have kept those directories.

### Generating Proofs

The code for generating Isabelle proofs for the data provided in `data.tsv` (which has the MATH problems in [miniF2F](https://github.com/facebookresearch/miniF2F)) is in `isabelle/generate.py`.
For example, to call GPT-4o-mini, with the default [DSP](https://openreview.net/forum?id=SMa9EAovKMC) prompt with 8-shot examples, 

At `isabelle/`:
```shell
$ python generate.py -prompt_type dsp -num_ex 8 -model_type chat -model gpt-4o-mini -temperature 0.5
```

This will create a new directory within `isabelle/results` that looks like `dsp-num-ex-8-*` which will store the model generated proofs in a `predictions.tsv` file.

Please look at the command line arguments in the `isabelle/generate.py` file to change the various experimental settings. For example, to generate proofs with Llama3.1-70B using the aliased descriptions-only prompt, do the following:

Host the Llama model locally using [vllm](https://github.com/vllm-project/vllm):

```shell
$ CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3.1-70B-Instruct --download-dir YOUR_DOWNLOAD_DIR --tensor-parallel-size 2 --rope-scaling='{"type": "extended", "factor": 8.0}' --max_model_len 4000 --gpu_memory_utilization 0.95
```

Then at `isabelle/`:
```shell
$ python generate.py -prompt_type alias_isabelle_desc -model_type vllm -model meta-llama/Meta-Llama-3.1-70B-Instruct
```

### Evaluating Proofs

First, start the PISA server,

At `Portal-to-ISAbelle/`:
```shell
$ sbt -Djava.security.manager=allow "runMain pisa.server.PisaOneStageServer9000"
```

The code for evaluating proofs is provided in `isabelle/evaluate.py`. In a separate window, execute this code by passing the previously generated directory as a command line argument,

At `isabelle/`:
```shell
$ python evaluate.py -progs_name dsp-num-ex-8-*
```

This will print the accuracy and store the results in the file `execution.tsv` in the same directory with the model generated proofs.


# Citation

If you use our data or code, please cite our work:

```
@inproceedings{patel-etal-2024-evaluating,
    title = "Evaluating In-Context Learning of Libraries for Code Generation",
    author = "Patel, Arkil  and
      Reddy, Siva  and
      Bahdanau, Dzmitry  and
      Dasigi, Pradeep",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.161",
    pages = "2908--2926",
    abstract = "Contemporary Large Language Models (LLMs) exhibit a high degree of code generation and comprehension capability. A particularly promising area is their ability to interpret code modules from unfamiliar libraries for solving user-instructed tasks. Recent work has shown that large proprietary LLMs can learn novel library usage in-context from demonstrations. These results raise several open questions: whether demonstrations of library usage is required, whether smaller (and more open) models also possess such capabilities, etc. In this work, we take a broader approach by systematically evaluating a diverse array of LLMs across three scenarios reflecting varying levels of domain specialization to understand their abilities and limitations in generating code based on libraries defined in-context. Our results show that even smaller open-source LLMs like Llama-2 and StarCoder demonstrate an adept understanding of novel code libraries based on specification presented in-context. Our findings further reveal that LLMs exhibit a surprisingly high proficiency in learning novel library modules even when provided with just natural language descriptions or raw code implementations of the functions, which are often cheaper to obtain than demonstrations. Overall, our results pave the way for harnessing LLMs in more adaptable and dynamic coding environments.",
}
```

For any clarification, comments, or suggestions please contact [Arkil](http://arkilpatel.github.io/).

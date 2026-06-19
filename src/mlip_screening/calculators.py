import os
from pathlib import Path


def build_calculator(config):
    model = config["model"].lower()
    device = config.get("device", "cuda")

    if model == "uma":
        return build_uma(config, device)
    if model == "mace":
        return build_mace(config, device)
    if model == "nequip":
        return build_nequip(config, device)
    if model == "sevennet":
        return build_sevennet(config)
    if model == "equiformer":
        return build_equiformer(config)
    if model == "orb":
        return build_orb(config, device)

    raise ValueError(f"Unsupported model: {model}")


def build_uma(config, device):
    from torch.serialization import add_safe_globals
    from huggingface_hub import hf_hub_download
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit

    add_safe_globals([slice])

    checkpoint_path = hf_hub_download(
        repo_id=config.get("checkpoint_repo", "facebook/UMA"),
        filename=config.get("checkpoint_file", "uma-s-1p1.pt"),
        subfolder=config.get("checkpoint_subfolder", "checkpoints"),
    )

    predictor = load_predict_unit(
        checkpoint_path,
        inference_settings=config.get("inference_settings", "default"),
        device=device,
    )

    return FAIRChemCalculator(predictor, task_name=config.get("task_name", "omat"))


def build_mace(config, device):
    from mace.calculators import mace_mp

    return mace_mp(
        model=config.get("model_name", "medium-omat-0"),
        device=device,
        default_dtype=config.get("default_dtype", "float64"),
    )


def build_nequip(config, device):
    from nequip.integrations.ase import NequIPCalculator

    model_path = config.get("model_path") or os.environ.get("NEQUIP_MODEL", "").strip()
    if not model_path:
        raise ValueError("Set model_path in configs/nequip.yaml or export NEQUIP_MODEL.")

    return NequIPCalculator.from_compiled_model(
        compile_path=model_path,
        device=device,
        chemical_species_to_atom_type_map=config.get("chemical_species_to_atom_type_map", True),
    )


def build_sevennet(config):
    from sevenn.calculator import SevenNetCalculator

    return SevenNetCalculator(model=config.get("model_name", "7net-omat"))


def build_equiformer(config):
    import os
    import sys
    from pathlib import Path

    import yaml
    from huggingface_hub import hf_hub_download

    repo_root = Path(os.path.expandvars(config["repo_root"])).expanduser().resolve()
    src_path = repo_root / "src"

    if not src_path.exists():
        raise FileNotFoundError(f"EquiformerV3 src directory not found: {src_path}")

    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    import fairchem.experimental.trainers.equiformer_v3_dens_trainer  # noqa: F401
    import fairchem.experimental.models.equiformer_v3.equiformer_v3  # noqa: F401
    from fairchem.core.common.relaxation.ase_utils import OCPCalculator

    checkpoint_path = hf_hub_download(
        repo_id=config.get("hf_repo_id", "mirror-physics/equiformer_v3"),
        filename=config.get(
            "hf_checkpoint_file",
            "checkpoint/omat24-mptrj-salex_gradient.pt",
        ),
    )

    original_config = Path(
        os.path.expandvars(config["original_config"])
    ).expanduser().resolve()

    if not original_config.exists():
        raise FileNotFoundError(f"EquiformerV3 config not found: {original_config}")

    cfg = yaml.safe_load(original_config.read_text())

    if "optim" in cfg:
        cfg["optim"].pop("load_pretrained_weights", None)

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

    work_dir = Path(config.get("work_dir", "results/equiformer/configs")).expanduser()
    work_dir.mkdir(parents=True, exist_ok=True)

    fixed_config = work_dir / f"{original_config.stem}_inference_task_{task_id:02d}.yml"
    fixed_config.write_text(yaml.safe_dump(cfg, sort_keys=False))

    return OCPCalculator(
        checkpoint_path=checkpoint_path,
        config_yml=str(fixed_config),
        cpu=bool(config.get("cpu", False)),
        seed=int(config.get("seed", 42)),
    )


def build_orb(config, device):
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator

    function_name = config.get("pretrained_function", "orb_v3_conservative_inf_omat")
    precision = config.get("precision", "float32-high")
    orb_factory = getattr(pretrained, function_name)
    orbff = orb_factory(device=device, precision=precision)

    return ORBCalculator(orbff, device=device)

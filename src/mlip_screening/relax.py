import time

from ase.io import read
from ase.optimize import LBFGS
from ase.filters import FrechetCellFilter


def relax_structure(path, calculator, fmax=0.02, steps=500):
    start_time = time.perf_counter()

    atoms = read(path)
    atoms.calc = calculator

    opt = LBFGS(FrechetCellFilter(atoms), logfile=None)
    converged = opt.run(fmax=fmax, steps=steps)

    energy = atoms.get_potential_energy()
    volume = atoms.get_volume()
    elapsed = time.perf_counter() - start_time

    return energy, volume, elapsed, converged


def safe_relax_structure(path, calculator, fmax=0.02, steps=500):
    try:
        energy, volume, elapsed, converged = relax_structure(
            path=path,
            calculator=calculator,
            fmax=fmax,
            steps=steps,
        )
        return energy, volume, elapsed, converged, "ok", ""
    except Exception as exc:
        return None, None, None, False, "failed", f"{type(exc).__name__}: {exc}"

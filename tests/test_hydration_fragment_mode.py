from rdkit import Chem

from knf_core import geometry
from knf_core.engine.types import RunOptions


def test_hydration_grouping_merges_explicit_waters_without_moving_atoms():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO.O.O"))
    raw_fragments = geometry.detect_fragments(mol)
    atom_numbers_before = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

    grouped, diagnostics = geometry.group_hydration_fragments(mol, raw_fragments)

    assert len(raw_fragments) == 3
    assert len(grouped) == 2
    assert diagnostics["active"] is True
    assert diagnostics["water_count"] == 2
    assert diagnostics["water_atom_count"] == 6
    assert set(grouped[0]).isdisjoint(grouped[1])
    assert sorted(grouped[0] + grouped[1]) == list(range(mol.GetNumAtoms()))
    assert [atom.GetAtomicNum() for atom in mol.GetAtoms()] == atom_numbers_before


def test_hydration_mode_is_part_of_knf_run_options():
    options = RunOptions(hydration_fragment_mode=True)
    assert options.hydration_fragment_mode is True

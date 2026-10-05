from unittest.mock import patch, MagicMock
from dataclasses import asdict
from backend.optimization import optimize
from backend.hemodynamics import ClipPlacement, ValveGeometry, SimulationResult

def test_optimize_prioritizes_recommended():
    geom = MagicMock(spec=ValveGeometry)
    
    # Res 1: Recommended, Objective 10.0
    res1 = MagicMock(spec=SimulationResult)
    res1.within_physiologic_range = True
    res1.regurgitation_ml = 0.5
    res1.to_dict.return_value = {"regurgitation_ml": 0.5}
    
    # Res 2: Review, Objective 5.0
    res2 = MagicMock(spec=SimulationResult)
    res2.within_physiologic_range = True
    res2.regurgitation_ml = 2.0  # exceeds 1.0
    res2.to_dict.return_value = {"regurgitation_ml": 2.0}
    
    # Res 3: Recommended, Objective 15.0
    res3 = MagicMock(spec=SimulationResult)
    res3.within_physiologic_range = True
    res3.regurgitation_ml = 0.8
    res3.to_dict.return_value = {"regurgitation_ml": 0.8}

    # Res 4: Review, Objective 2.0
    res4 = MagicMock(spec=SimulationResult)
    res4.within_physiologic_range = False # fails physiologic range
    res4.regurgitation_ml = 5.0
    res4.to_dict.return_value = {"regurgitation_ml": 5.0}

    # Res 5: Recommended, Objective 10.0 (tie with Res 1)
    res5 = MagicMock(spec=SimulationResult)
    res5.within_physiologic_range = True
    res5.regurgitation_ml = 0.5
    res5.to_dict.return_value = {"regurgitation_ml": 0.5}

    with patch('backend.optimization._grid') as mock_grid:
        # 5 clips, positions in [0, 1]
        clips = [ClipPlacement(i/4, 4.0, 0.0, 1) for i in range(5)]
        mock_grid.return_value = clips
        
        with patch('backend.optimization.simulate') as mock_simulate:
            # Return them in order: res4, res2, res1, res5, res3
            mock_simulate.side_effect = [res4, res2, res1, res5, res3]
            
            with patch('backend.optimization.objective') as mock_objective:
                mock_objective.side_effect = [2.0, 5.0, 10.0, 10.0, 15.0]
                
                # Test top_k = 5
                ranked = optimize(geom, top_k=5)
                
                # Expected order:
                # Recommended (sorted by obj):
                # 1. Res 1 (10.0) - clips[2]
                # 2. Res 5 (10.0) - clips[3]
                # 3. Res 3 (15.0) - clips[4]
                # Review (sorted by obj):
                # 4. Res 4 (2.0)  - clips[0]
                # 5. Res 2 (5.0)  - clips[1]

                assert len(ranked) == 5
                # Res 1
                assert ranked[0]["recommendation"] == "Recommended"
                assert ranked[0]["objective"] == 10.0
                assert ranked[0]["placement"] == asdict(clips[2])
                # Res 5
                assert ranked[1]["recommendation"] == "Recommended"
                assert ranked[1]["objective"] == 10.0
                assert ranked[1]["placement"] == asdict(clips[3])
                # Res 3
                assert ranked[2]["recommendation"] == "Recommended"
                assert ranked[2]["objective"] == 15.0
                assert ranked[2]["placement"] == asdict(clips[4])
                # Res 4
                assert ranked[3]["recommendation"] == "Review"
                assert ranked[3]["objective"] == 2.0
                assert ranked[3]["placement"] == asdict(clips[0])
                # Res 2
                assert ranked[4]["recommendation"] == "Review"
                assert ranked[4]["objective"] == 5.0
                assert ranked[4]["placement"] == asdict(clips[1])

def test_optimize_top_k_behavior():
    geom = MagicMock(spec=ValveGeometry)
    
    # Res 1: Recommended, Obj 10.0
    res1 = MagicMock(spec=SimulationResult)
    res1.within_physiologic_range = True
    res1.regurgitation_ml = 0.5
    res1.to_dict.return_value = {"regurgitation_ml": 0.5}
    
    # Res 2: Review, Obj 2.0
    res2 = MagicMock(spec=SimulationResult)
    res2.within_physiologic_range = False
    res2.regurgitation_ml = 5.0
    res2.to_dict.return_value = {"regurgitation_ml": 5.0}

    with patch('backend.optimization._grid') as mock_grid:
        mock_grid.return_value = [ClipPlacement(1, 4.0, 0.0, 1), ClipPlacement(2, 4.0, 0.0, 1)]
        with patch('backend.optimization.simulate') as mock_simulate:
            mock_simulate.side_effect = [res1, res2]
            with patch('backend.optimization.objective') as mock_objective:
                mock_objective.side_effect = [10.0, 2.0]
                
                # top_k = 1
                ranked = optimize(geom, top_k=1)
                assert len(ranked) == 1
                # Should be the recommended one even if objective is higher
                assert ranked[0]["recommendation"] == "Recommended"
                assert ranked[0]["objective"] == 10.0

def test_optimize_threshold_edge_case():
    geom = MagicMock(spec=ValveGeometry)
    
    # Res 1: Recommended (threshold 1.0)
    res1 = MagicMock(spec=SimulationResult)
    res1.within_physiologic_range = True
    res1.regurgitation_ml = 1.0
    res1.to_dict.return_value = {"regurgitation_ml": 1.0}
    
    # Res 2: Review (just above 1.0)
    res2 = MagicMock(spec=SimulationResult)
    res2.within_physiologic_range = True
    res2.regurgitation_ml = 1.001
    res2.to_dict.return_value = {"regurgitation_ml": 1.001}

    with patch('backend.optimization._grid') as mock_grid:
        mock_grid.return_value = [ClipPlacement(0.5, 4.0, 0.0, 1), ClipPlacement(0.6, 4.0, 0.0, 1)]
        with patch('backend.optimization.simulate') as mock_simulate:
            mock_simulate.side_effect = [res1, res2]
            with patch('backend.optimization.objective') as mock_objective:
                mock_objective.side_effect = [10.0, 5.0]
                
                ranked = optimize(geom, top_k=2)
                assert ranked[0]["recommendation"] == "Recommended"
                assert ranked[1]["recommendation"] == "Review"

def test_optimize_all_review():
    geom = MagicMock(spec=ValveGeometry)
    
    # All are Review
    res1 = MagicMock(spec=SimulationResult)
    res1.within_physiologic_range = False
    res1.regurgitation_ml = 0.5
    res1.to_dict.return_value = {"regurgitation_ml": 0.5}
    
    res2 = MagicMock(spec=SimulationResult)
    res2.within_physiologic_range = True
    res2.regurgitation_ml = 2.0
    res2.to_dict.return_value = {"regurgitation_ml": 2.0}

    with patch('backend.optimization._grid') as mock_grid:
        mock_grid.return_value = [ClipPlacement(0.5, 4.0, 0.0, 1), ClipPlacement(0.6, 4.0, 0.0, 1)]
        with patch('backend.optimization.simulate') as mock_simulate:
            mock_simulate.side_effect = [res1, res2]
            with patch('backend.optimization.objective') as mock_objective:
                mock_objective.side_effect = [10.0, 5.0]
                
                # Both are Review. Sorted by objective: res2 (5.0), then res1 (10.0)
                ranked = optimize(geom, top_k=2)
                assert ranked[0]["recommendation"] == "Review"
                assert ranked[0]["objective"] == 5.0
                assert ranked[1]["recommendation"] == "Review"
                assert ranked[1]["objective"] == 10.0

def test_optimize_top_k_zero():
    geom = MagicMock(spec=ValveGeometry)
    with patch('backend.optimization._grid') as mock_grid:
        mock_grid.return_value = [ClipPlacement(0.5, 4.0, 0.0, 1)]
        with patch('backend.optimization.simulate') as mock_simulate:
            res = MagicMock(spec=SimulationResult)
            res.within_physiologic_range = True
            res.regurgitation_ml = 0.5
            mock_simulate.return_value = res
            with patch('backend.optimization.objective') as mock_objective:
                mock_objective.return_value = 10.0
                ranked = optimize(geom, top_k=0)
                assert ranked == []


def _adversarial_ranking(monkeypatch, top_k):
    """Shuffled objectives and descending positions expose unstable tie keys."""
    from backend import optimization
    clips = [ClipPlacement(pos, 4.0, 0.0, 1) for pos in (0.7, 0.6, 0.4, 0.3, 0.2, 0.1)]
    recommended = (True, True, False, True, False, False)
    results = [SimulationResult(0.5 if rec else 2.0, 3.0, 100.0, 1.0, True) for rec in recommended]
    scores = dict(zip(map(id, results), (15.0, 10.0, 3.0, 10.0, 3.0, 1.0)))
    monkeypatch.setattr(optimization, "_grid", lambda: iter(clips))
    monkeypatch.setattr(optimization, "simulate", lambda geom, clip: results[clips.index(clip)])
    monkeypatch.setattr(optimization, "objective", lambda result: scores[id(result)])
    return optimization.optimize(MagicMock(spec=ValveGeometry), top_k=top_k)


def test_shuffled_objectives_and_stable_ties_in_both_categories(monkeypatch):
    ranked = _adversarial_ranking(monkeypatch, top_k=6)
    assert [r["placement"]["position"] for r in ranked] == [0.6, 0.3, 0.7, 0.1, 0.4, 0.2]
    assert [r["objective"] for r in ranked] == [10.0, 10.0, 15.0, 1.0, 3.0, 3.0]
    assert [r["recommendation"] for r in ranked] == ["Recommended"] * 3 + ["Review"] * 3


def test_sorting_precedes_top_k_truncation(monkeypatch):
    ranked = _adversarial_ranking(monkeypatch, top_k=1)
    assert len(ranked) == 1
    # Best Recommended candidate is outside the original one-item prefix.
    assert ranked[0]["placement"]["position"] == 0.6
    assert ranked[0]["objective"] == 10.0

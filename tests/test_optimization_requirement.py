import pytest
from unittest.mock import patch, MagicMock
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
        # 5 clips
        clips = [ClipPlacement(i, 4.0, 0.0, 1) for i in range(5)]
        mock_grid.return_value = clips
        
        with patch('backend.optimization.simulate') as mock_simulate:
            # Return them in objective order: 2.0, 5.0, 10.0, 10.0, 15.0
            mock_simulate.side_effect = [res4, res2, res1, res5, res3]
            
            with patch('backend.optimization.objective') as mock_objective:
                mock_objective.side_effect = [2.0, 5.0, 10.0, 10.0, 15.0]
                
                # Test top_k = 5
                ranked = optimize(geom, top_k=5)
                
                # EXPECTED (with new logic):
                # Recommended (sorted by obj):
                # 1. Res 1 (10.0)
                # 2. Res 5 (10.0)
                # 3. Res 3 (15.0)
                # Review (sorted by obj):
                # 4. Res 4 (2.0)
                # 5. Res 2 (5.0)
                
                assert len(ranked) == 5
                assert ranked[0]["recommendation"] == "Recommended"
                assert ranked[0]["objective"] == 10.0
                assert ranked[1]["recommendation"] == "Recommended"
                assert ranked[1]["objective"] == 10.0
                assert ranked[2]["recommendation"] == "Recommended"
                assert ranked[2]["objective"] == 15.0
                assert ranked[3]["recommendation"] == "Review"
                assert ranked[3]["objective"] == 2.0
                assert ranked[4]["recommendation"] == "Review"
                assert ranked[4]["objective"] == 5.0

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


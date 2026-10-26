def test_dashboard_import_does_not_start_training() -> None:
    import app

    assert callable(app.main)

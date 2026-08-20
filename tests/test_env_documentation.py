from scripts.check_env_documentation import undocumented_environment_names


def test_source_referenced_environment_variables_are_documented():
    assert undocumented_environment_names() == []

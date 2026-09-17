# Release procedure

## Preparing the code

###  For major and minor release.
- Create a new branch `qutip-jax-X.Y.0` with the code at the state of the release.
- Create a PR to update the version number and changelog.

###  For micro release.
- Cherry-pick the change to bring to the release and create a PR for the minor version branch.
- Update the version number and changelog. (Can be the same PR or a different on to the previous one.)

**Don't forget to push to changelog to the master branch also.**

## Make the git release

- Create the tag and fill with the changelog.
- This will automatically trigger the action to release to pypi and also add the wheels to the GitHub release.

## Post-release ToDo's

1. In [qutip-tutorials](https://github.com/qutip/qutip-tutorials), upgrade `qutip-jax` dependency for tests with the released `qutip` by updating the value in `jax-install-spec` keyword of the 'v5-release' job.
The following workflows have to be modified:
    1. `.github/workflows/notebook_ci.yaml`
    2. `.github/workflows/nightly_ci.yaml`

## Readthedocs

- Readthedocs should build the documentation for the new tag automatically. Can take some time. If it does not catch the release, trigger the build manually.

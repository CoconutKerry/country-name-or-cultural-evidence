# Public release checklist

Complete these items after anonymous review and before making the repository
public:

- replace anonymous metadata with the final author list and affiliations;
- add the final paper citation, DOI or proceedings URL, and repository URL;
- select and add a repository-wide code license;
- review redistribution terms for GlobalOpinionQA, Pew, and WVS-derived data;
- review each GGUF and upstream model license;
- confirm that no model weights, cache files, personal paths, credentials, or
  private review material are present;
- decide whether full prompt text may be redistributed with the processed data;
- run `make reproduce`, `make test`, and `make verify` in a clean environment;
- regenerate `provenance/RELEASE_SHA256SUMS.txt` after any change; and
- archive the public release in a persistent repository if a DOI is required.

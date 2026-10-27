{
  description = "anilist-recommender — local AniList anime/manga recommendation CLI (Ollama + sqlite-vec + sentence-transformers on CPU)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;

      # torch / sentence-transformers / sqlite-vec manylinux wheels expect a FHS
      # layout. On NixOS those shared objects cannot find libstdc++ / libgcc /
      # libz on their own, so we expose them via LD_LIBRARY_PATH at runtime.
      runtimeLibs = pkgs: [ pkgs.stdenv.cc.cc.lib pkgs.zlib ];

      # A runnable `anilist-rec` binary: builds the project + its PyPI deps with
      # `uv` straight into $out, then wraps `python -m anilist_recommender`.
      #
      # NOTE: this fetches PyPI wheels (torch / sentence-transformers / sqlite-vec)
      # and therefore needs NETWORK access at build time (see README).
      mkApp = pkgs:
        let
          python = pkgs.python312;
        in
        pkgs.stdenv.mkDerivation {
          pname = "anilist-recommender";
          version = "0.1.0";
          src = self;

          nativeBuildInputs = [ pkgs.uv python pkgs.makeWrapper ];

          installPhase = ''
            runHook preInstall
            mkdir -p "$out/bin" "$out/${python.sitePackages}"
            uv pip install \
              --python "${python}/bin/python3" \
              --target "$out/${python.sitePackages}" \
              --no-cache \
              .
            makeWrapper "${python}/bin/python3" "$out/bin/anilist-rec" \
              --add-flags "-m anilist_recommender" \
              --set PYTHONPATH "$out/${python.sitePackages}" \
              --set LD_LIBRARY_PATH "${pkgs.lib.makeLibraryPath (runtimeLibs pkgs)}"
            runHook postInstall
          '';
        };
    in
    {
      packages = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = mkApp pkgs;
          anilist-recommender = mkApp pkgs;
        });

      apps = forAllSystems (system: {
        default = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/anilist-rec";
        };
      });

      devShells = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              uv          # Python package manager (resolves & installs the venv)
              sqlite      # sqlite3 CLI for inspecting data/anilist_local.db
              python312   # interpreter used by uv to build the .venv
              curl        # handy for smoke-testing the Ollama endpoint
            ];

            shellHook = ''
              export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath (runtimeLibs pkgs)}:''${LD_LIBRARY_PATH:-}
              echo "📚 anilist-recommender — run: uv sync && uv run anilist-rec --help"
            '';
          };
        });
    };
}

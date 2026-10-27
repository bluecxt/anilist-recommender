{
  description = "anilist-recommender — local AniList anime/manga recommendation CLI (Ollama + sqlite-vec + sentence-transformers on CPU)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;

      # Needed only by the devShell: `uv sync` installs PyPI manylinux wheels
      # (torch) that cannot find libstdc++/libz on NixOS by themselves. The
      # `packages.default` build uses nixpkgs' own (correctly patched) torch.
      runtimeLibs = pkgs: [ pkgs.stdenv.cc.cc.lib pkgs.zlib ];
    in
    {
      packages = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          pythonPackages = pkgs.python313Packages;
          app = pythonPackages.buildPythonApplication {
            pname = "anilist-recommender";
            version = "0.1.0";
            pyproject = true;
            src = self;

            build-system = with pythonPackages; [ hatchling ];

            dependencies = with pythonPackages; [
              httpx
              pydantic
              openai
              typer
              rich
              sqlite-vec
              sentence-transformers
            ];

            meta = {
              description = "Local AniList anime/manga recommender (Ollama + sqlite-vec)";
              mainProgram = "anilist-rec";
              license = nixpkgs.lib.licenses.mit;
            };
          };
        in
        {
          default = app;
          anilist-recommender = app;
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
              python313   # interpreter used by uv to build the .venv
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

{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    flake-parts.url = "github:hercules-ci/flake-parts";
    flakever.url = "github:numinit/flakever";
    treefmt-nix.url = "github:numtide/treefmt-nix";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-parts,
      flakever,
      treefmt-nix,
      ...
    }@inputs:
    let
      flakeverConfig = flakever.lib.mkFlakever {
        inherit inputs;

        digits = [
          1
          2
          2
        ];
      };
    in
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
        inputs.flake-parts.flakeModules.easyOverlay
        inputs.treefmt-nix.flakeModule
      ];

      flake.versionTemplate = "1.1pre-<lastModifiedDate>-<rev>";

      systems = [
        "aarch64-linux"
        "x86_64-linux"
      ];

      perSystem =
        {
          system,
          pkgs,
          ...
        }:
        let
          inherit (pkgs) lib;
        in
        {
          _module.args.pkgs = import inputs.nixpkgs {
            inherit system;
            overlays = [
              self.overlays.default
            ];
          };

          treefmt.programs = {
            nixfmt.enable = true;
            zig.enable = true;
          };

          overlayAttrs = {
            flakever = flakeverConfig;
            erbium-hal = pkgs.callPackage ./pkgs/erbium-hal { };
            etsoc-sysemu = pkgs.callPackage ./pkgs/etsoc-sysemu { };
            vulcan = pkgs.callPackage ./pkgs/vulcan { };
          };

          packages = {
            default = pkgs.vulcan;
            inherit (pkgs) erbium-hal etsoc-sysemu;
          };

          devShells.default = pkgs.vulcan.shell;

          checks = {
            default = pkgs.vulcan;
            inherit (pkgs) erbium-hal etsoc-sysemu;
          };
        };
    };
}

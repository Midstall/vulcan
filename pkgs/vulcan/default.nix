{
  lib,
  flakever,
  stdenv,
  mkShell,
  zig,
  lld,
  binutils,
}:
stdenv.mkDerivation (finalAttrs: {
  pname = "vulcan";
  inherit (flakever) version;

  src = lib.cleanSource ../../.;

  nativeBuildInputs = [ zig ];

  deps = zig.fetchDeps {
    inherit (finalAttrs) pname version src;
    hash = "sha256-S9/I0CRLDbeYxhXhrhPSoHTSLLVTqpVLwE7D/Lbwhvg=";
  };

  postConfigure = ''
    ln -s $deps $ZIG_GLOBAL_CACHE_DIR/p
  '';

  passthru.shell = mkShell {
    name = "vulcan-dev-shell";

    packages = [
      zig
      lld
      binutils
    ];
  };
})

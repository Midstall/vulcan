{
  lib,
  flakever,
  stdenv,
  mkShell,
  wrapCCWith,
  zig,
  lld,
  binutils,
  etsoc-sysemu,
}:
let
  inherit (stdenv) targetPlatform hostPlatform;

  targetPrefix = lib.optionalString (targetPlatform != hostPlatform) (targetPlatform.config + "-");
in
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

  doCheck = true;
  nativeCheckInputs = lib.optional etsoc-sysemu.meta.available etsoc-sysemu;

  # Needed so `wrapCCWith` works, remove when nixpkgs supports `vcc`
  postInstall = ''
    ln -s $out/bin/vcc $out/bin/${targetPrefix}gcc
  '';

  passthru = {
    shell = mkShell {
      name = "vulcan-dev-shell";

      packages = [
        zig
        lld
        binutils
      ];
    };

    cc = wrapCCWith {
      cc = finalAttrs.finalPackage;
    };

    vcc-stdenv = stdenv.override {
      inherit (finalAttrs.finalPackage) cc;
      allowedRequisites = stdenv.allowedRequisites ++ [
        finalAttrs.finalPackage
        finalAttrs.finalPackage.cc.expand-response-params
        finalAttrs.finalPackage.cc.bintools
      ];
    };
  };
})

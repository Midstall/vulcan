{
  lib,
  stdenv,
  fetchFromGitHub,
  cmake,
  glog,
  lz4,
  erbium-hal,
}:
stdenv.mkDerivation (finalAttrs: {
  pname = "etsoc-sysemu";

  inherit (erbium-hal) version src;

  sourceRoot = "source/sw-sysemu";

  nativeBuildInputs = [ cmake ];
  buildInputs = [
    erbium-hal
    glog
    lz4
  ];

  postPatch = ''
    substituteInPlace CMakeLists.txt \
      --replace-fail "-pedantic-errors -Werror -Wno-implicit-fallthrough" "-pedantic-errors -Wno-implicit-fallthrough"
  '';

  cmakeFlags = [
    (lib.cmakeFeature "CMAKE_BUILD_TYPE" "Release")
    (lib.cmakeBool "BENCHMARKS" false)
    (lib.cmakeBool "BACKTRACE" false)
    (lib.cmakeBool "PRELOAD_LZ4" false)
  ];

  buildFlags = [ "sys_emu" ];

  installPhase = ''
    runHook preInstall
    install -Dm755 "$(find . -name sys_emu -type f -perm -u+x | head -n1)" -t "$out/bin"
    runHook postInstall
  '';

  meta = erbium-hal.meta // {
    description = "ETSOC-1 (CORE-ET Erbium) functional interpreter";
    mainProgram = "sys_emu";
  };
})

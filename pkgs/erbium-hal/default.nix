{
  lib,
  stdenv,
  fetchFromGitHub,
  cmake,
}:
stdenv.mkDerivation (finalAttrs: {
  pname = "erbium-hal";
  version = "0.1.0";

  src = fetchFromGitHub {
    owner = "aifoundry-org";
    repo = "et-platform";
    rev = "5ec553742455905e8819b2acc91278c208236ccd";
    hash = "sha256-ZIxcSbJoqRdAhefauxnSenPcUSsWuuxTOm5l4eWWSUA=";
  };

  sourceRoot = "source/erbium-hal";

  nativeBuildInputs = [ cmake ];

  meta = {
    description = "CORE-ET Erbium hardware abstraction layer";
    homepage = "https://github.com/aifoundry-org/et-platform";
    license = lib.licenses.asl20;
    platforms = lib.platforms.all;
  };
})

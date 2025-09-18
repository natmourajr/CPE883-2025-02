{
  description = "A custom TeX Live environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }: let
    # --- IMPORTANT ---
    # Define your system here.
    # "x86_64-linux" is for standard Intel/AMD Linux.
    # Use "aarch64-linux" for ARM-based Linux (e.g., Raspberry Pi 4).
    # Use "x86_64-darwin" for Intel-based Macs.
    # Use "aarch64-darwin" for Apple Silicon Macs (M1/M2/M3).
    system = "x86_64-linux";

    # Get the package set for the specified system.
    pkgs = nixpkgs.legacyPackages.${system};

  in {
    # Define our custom package for the specified system.
    packages.${system}.my-texlive = pkgs.texlive.combined.scheme-medium.withPackages (ps: [
      ps.caladea
      ps.carlito
      ps.fontaxes
      ps.yfonts
      ps.collection-fontsrecommended
    ]);
  };
}

declare module "iwanthue-api" {
  interface IWantHueColor {
    rgb: number[]
    hcl(): number[]
    lab(): number[]
  }

  interface IWantHueApi {
    generate(
      colorsCount: number,
      checkColor?: (color: IWantHueColor) => boolean,
      forceMode?: boolean,
      quality?: number,
      ultra_precision?: boolean,
    ): IWantHueColor[]
    diffSort(colors: IWantHueColor[]): IWantHueColor[]
  }

  function createIWantHue(): IWantHueApi
  export default createIWantHue
}

export type APIResult = SuccessfulResult | FailedResult;

export type ImageProbabilities = [string, number][]

export interface SuccessfulResult {
    success: true
    data: ImageProbabilities[]
}

export interface FailedResult {
    success: false
}
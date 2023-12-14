export function getBase64(file: File): Promise<string> {
    return new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = function () {
            resolve(reader.result as string)
        };
        reader.onerror = function (error) {
            reject(error)
        };
    })
}
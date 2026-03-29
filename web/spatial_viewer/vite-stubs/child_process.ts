function noopStream() {
  return { on: () => {} };
}

export function spawn(
  _command: string,
  _args?: readonly string[],
  _options?: object,
) {
  return {
    stdout: noopStream(),
    stderr: noopStream(),
    on: (event: string, cb: (code?: number) => void) => {
      if (event === "close") {
        queueMicrotask(() => cb(0));
      }
    },
    kill: () => {},
  };
}

export default { spawn };

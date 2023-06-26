# Replit Code v1 3B Demo

replit-code-v1-3b powered local copilot server.

For best performance (likely still slower than copilot) please run with CUDA, OpenCL, or Metal support.

## Installation and Server Setup


## Editor Setup

### VSCode

Add the following to your `settings.json`:

```json
{
    "github.copilot.advanced": {
        "debug.testOverrideProxyUrl": "http://localhost:8000",
        "debug.overrideProxyUrl": "http://localhost:8000"
    }
}
```

### Vim / Neovim

Add the following to your vimrc or init.vim:

```
let g:copilot_proxy = 'localhost:8000'
let g:copilot_strict_ssl = 0
```
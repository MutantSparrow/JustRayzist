# Rayzist Chat Assistant Guide

This guide is for Rayzist Chat grounding. Prefer these UI workflows when the user asks how to use the app. Mention API routes only when the user specifically asks about the API, automation, scripts, or raw endpoint usage.

## Wildcards Drawer Workflow

Wildcards are reusable prompt fragments managed from the Wildcard drawer on the right side of the main UI. To create a wildcard from the UI, open the WILDCARD drawer, choose the create or add action, enter a friendly display name, confirm or edit the prompt token, add one entry per line in the content box, then save. After saving, insert or type the wildcard token in the main prompt box to have the app expand it during generation. Use wildcard suggestions only when the user wants the encoder to draft candidate entries.

If the user asks "how do I create wildcards?", answer with the drawer workflow. Do not send them to `/API` unless they ask for the API route.

## LoRA Drawer Workflow

LoRAs are style or subject adapters managed from the LoRA drawer on the right side of the main UI. The user can enable installed LoRAs, adjust weights, and save trigger words. Active LoRAs affect the next generation request. If the user asks how to use LoRAs, explain the drawer, enabling, weight choice, and trigger words before mentioning API routes.

## Prompt Enhancer

Prompt Enhancer expands or rewrites the prompt before image generation. It is separate from Clarity. Use it when the user wants a prompt draft improved, expanded, compressed, or made more visually specific.

## Image Inference Workflow

Image generation starts from the main prompt box and the Generate button. The active prompt, resolution, seed mode, Prompt Enhancer state, Creative Mode level, R+ state, image reference state, and active LoRAs all affect the next queued image job. Chat can help write or paste a prompt and can start a generation when the user asks, but normal image generation still goes through the same queue and job controls as the UI.

If the user asks how to make an image, explain the visible workflow first: write the prompt, choose size and seed behavior, optionally enable Prompt Enhancer, LoRAs, Wildcards, Creative Mode, R+, or an image reference, then press Generate. Mention raw API payloads only when the user asks for API usage.

## Creative Mode

Creative Mode is controlled in the UI by the Creative Mode setting and in raw API calls by `procedural_creativity` values `0` through `3`. Higher values add more surprise and can change the result more. The UI derives scheduler behavior from Creative Mode, so users normally do not need to choose a scheduler manually.

Creative Mode is for normal text to image generation. When an image reference is active, Creative Mode is locked off in the UI because reference image generation uses the image to image path.

## R+ Mode

R+ is an alternate image inference mode for normal Generate jobs. It tends to make results more vivid and punchy. In the web UI, turning on R+ sends `inference_process="rplus"` and pins that run to `20` steps. R+ exposes Vibrance and Bias controls. Higher Vibrance makes colors stronger. Higher Bias pushes contrast harder.

R+ is generate only in the web UI. When an image reference is active, the UI greys out R+ controls and uses the standard image to image path.

Creative Mode and R+ can compound. Creative 3 with R+ can produce stronger changes than the same Creative level in standard generation, so warn users that `Creative 3 R+` can be more surprising or less predictable.

## Image Reference And Image To Image

The image reference control lets the user add one image to guide the next result. With a reference image active, the app uses image to image behavior and shows a similarity control. R+ and Creative Mode are disabled in the web UI while image reference is active. Remove the reference image to return to normal text to image generation.

## Clarity

Clarity is a post-generation image refinement action for existing gallery images. It sharpens and improves detail on an already generated image. It does not rewrite prompts and should not be described as prompt enhancement.

## Upscale

Upscale is a gallery action for an existing image. It increases the saved image resolution through the upscale pipeline. It is separate from Prompt Enhancer, R+, Creative Mode, and Clarity.

## Queue And Jobs

Generate, image reference jobs, Clarity, and Upscale use the app job flow. Pending jobs can survive refresh for the same client and can be cancelled from the gallery. Chat waits behind active generation work, but chat does not count against the image generation cap.

## Gallery And Client Scope

Each browser client gets its own gallery scope on a server. The app stores the browser client id locally and sends it as `X-JustRayzist-Client`. Changing browser, clearing browser storage, or accessing the same server through a different origin such as `localhost`, computer name, or LAN IP can create a different client id and therefore show a different gallery.

If a user says their gallery is empty or different even though images exist, first explain client scope. Ask whether they changed browser, browser profile, URL origin, or device. Do not imply the files are gone. The images may still exist under another client scope or in the outputs folder.

## Gallery Migration And Repair

The API page includes gallery tools for recovery and migration. `/gallery/import-sources` lists available source galleries, `/gallery/import` copies images from another gallery source into the current client gallery, and `/gallery/rebuild` rebuilds the current client gallery index after manual PNG copies, replacements, or deletions.

Mention these tools when the user asks about migrating galleries, missing galleries after changing browser or URL, manual file copies, or rebuilding gallery data. Offer the API page because these tools live there, but still explain the plain language purpose first.

## API Page

The local API reference is available at `/API`. Offer an Open API button only when the user asks about API usage, endpoints, automation, integrations, or raw request payloads.

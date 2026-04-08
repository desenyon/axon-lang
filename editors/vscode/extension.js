/**
 * Axon Language Extension for Visual Studio Code
 * ================================================
 * Launches the Axon LSP server (axon lsp) and connects it to VS Code
 * using vscode-languageclient.
 *
 * The server communicates over stdin/stdout using JSON-RPC 2.0 (standard
 * LSP transport).
 */

"use strict";

const vscode = require("vscode");
const path = require("path");
const { LanguageClient, TransportKind } = require("vscode-languageclient/node");

/** @type {LanguageClient | undefined} */
let client;

/** Output channel for server messages */
let outputChannel;

/**
 * Build the server options from VS Code settings.
 * @returns {import('vscode-languageclient').ServerOptions}
 */
function buildServerOptions() {
  const config = vscode.workspace.getConfiguration("axon");
  const serverPath = config.get("serverPath", "axon");
  const serverArgs = config.get("serverArgs", ["lsp"]);

  return {
    command: serverPath,
    args: serverArgs,
    transport: TransportKind.stdio,
  };
}

/**
 * Build the language client options.
 * @returns {import('vscode-languageclient').LanguageClientOptions}
 */
function buildClientOptions() {
  return {
    documentSelector: [{ scheme: "file", language: "axon" }],
    synchronize: {
      fileEvents: vscode.workspace.createFileSystemWatcher("**/*.axon"),
    },
    outputChannel: outputChannel,
  };
}

/**
 * Start the language client.
 * @returns {LanguageClient}
 */
function startClient() {
  const serverOptions = buildServerOptions();
  const clientOptions = buildClientOptions();

  const newClient = new LanguageClient(
    "axon",
    "Axon Language Server",
    serverOptions,
    clientOptions
  );

  newClient.start();
  return newClient;
}

/**
 * Called when the extension is activated (first .axon file opened).
 * @param {vscode.ExtensionContext} context
 */
async function activate(context) {
  outputChannel = vscode.window.createOutputChannel("Axon Language Server");
  context.subscriptions.push(outputChannel);

  outputChannel.appendLine("Axon Language Extension activating...");

  // Register commands
  context.subscriptions.push(
    vscode.commands.registerCommand("axon.restartServer", async () => {
      outputChannel.appendLine("Restarting Axon Language Server...");
      if (client) {
        await client.stop();
      }
      client = startClient();
      vscode.window.showInformationMessage("Axon Language Server restarted.");
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("axon.showOutput", () => {
      outputChannel.show(true);
    })
  );

  // Start the server
  try {
    client = startClient();
    context.subscriptions.push(client);
    outputChannel.appendLine("Axon Language Server started.");
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    outputChannel.appendLine(`Failed to start Axon Language Server: ${msg}`);
    vscode.window.showErrorMessage(
      `Failed to start Axon Language Server: ${msg}. ` +
        "Make sure 'axon' is installed and on your PATH."
    );
  }
}

/**
 * Called when the extension is deactivated.
 */
async function deactivate() {
  if (client) {
    await client.stop();
    client = undefined;
  }
}

module.exports = { activate, deactivate };

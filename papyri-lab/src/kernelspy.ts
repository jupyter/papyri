// This module implement an object that spies on the kernel messages
// to intercept documentation print request.
import { VDomModel } from '@jupyterlab/apputils';
import { INotebookTracker, NotebookPanel } from '@jupyterlab/notebook';
import { Kernel, KernelMessage } from '@jupyterlab/services';
import { Signal, ISignal } from '@lumino/signaling';

export type MessageThread = {
  args: Kernel.IAnyMessageArgs;
  children: MessageThread[];
};

function isHeader(
  candidate: { [key: string]: undefined } | KernelMessage.IHeader
): candidate is KernelMessage.IHeader {
  return candidate.msg_id !== undefined;
}

/**
 * Model for a kernel spy.
 */
export class KernelSpyModel extends VDomModel {
  // constructor(kernel?: Kernel.IKernelConnection | null) {
  constructor(notebookTracker: INotebookTracker, path?: string) {
    console.log('notebook tracker constructed');
    super();
    this._notebookTracker = notebookTracker;
    this._notebookTracker.currentChanged.connect(this.onNotebookChanged, this);
    this.onNotebookChanged(undefined, { path });
  }

  onNotebookChanged(sender: any, args: any) {
    console.log('notebook changed', args);
    // onNotebookChanged(notebookTracker: INotebookTracker, path?: string) {
    if (args.path) {
      this._notebook =
        this._notebookTracker.find(nb => nb.context.path === args.path) ?? null;
    } else {
      this._notebook = this._notebookTracker.currentWidget;
    }

    if (this._notebook) {
      this.kernel =
        this._notebook.context.sessionContext?.session?.kernel ?? null;
      this._notebook.context.sessionContext.kernelChanged.connect((_, args) => {
        this.kernel = args.newValue;
      });
    } else {
      this.kernel = null;
    }
  }

  clear() {
    this._log.splice(0, this._log.length);
    this._messages = {};
    this._childLUT = {};
    this._roots = [];
    this.stateChanged.emit(void 0);
  }

  get kernel() {
    return this._kernel;
  }

  set kernel(value: Kernel.IKernelConnection | null) {
    if (this._kernel) {
      this._kernel.anyMessage.disconnect(this.onMessage, this);
    }
    this._kernel = value;
    if (this._kernel) {
      this._kernel.anyMessage.connect(this.onMessage, this);
    }
  }

  get log(): ReadonlyArray<Kernel.IAnyMessageArgs> {
    return this._log;
  }

  get tree(): MessageThread[] {
    return this._roots.map(rootId => {
      return this.getThread(rootId, false);
    });
  }

  depth(args: Kernel.IAnyMessageArgs | null): number {
    if (args === null) {
      return -1;
    }
    let depth = 0;
    while ((args = this._findParent(args))) {
      ++depth;
    }
    return depth;
  }

  getThread(msgId: string, ancestors = true): MessageThread {
    const args = this._messages[msgId];
    if (ancestors) {
      // Work up to root, then work downwards
      let root = args;
      let candidate;
      while ((candidate = this._findParent(root))) {
        root = candidate;
      }
      return this.getThread(root.msg.header.msg_id, false);
    }

    const childMessages = this._childLUT[msgId] || [];
    const childThreads = childMessages.map(childId => {
      return this.getThread(childId, false);
    });
    const thread: MessageThread = {
      args: this._messages[msgId],
      children: childThreads
    };
    return thread;
  }

  get questionMarkSubmitted(): ISignal<KernelSpyModel, string> {
    return this._questionMarkSubmitted;
  }

  protected onMessage(
    sender: Kernel.IKernelConnection,
    args: Kernel.IAnyMessageArgs
  ) {
    const { msg } = args;
    this._log.push(args);
    this._messages[msg.header.msg_id] = args;
    const parent = this._findParent(args);
    if (parent === null) {
      this._roots.push(msg.header.msg_id);
    } else {
      const header = parent.msg.header;
      this._childLUT[header.msg_id] = this._childLUT[header.msg_id] || [];
      this._childLUT[header.msg_id].push(msg.header.msg_id);
    }

    // Log the kernel message here.
    if (args.direction === 'recv') {
      const msg: any = args.msg;
      if (
        msg.channel === 'shell' &&
        msg.content !== undefined &&
        msg.content.payload !== undefined &&
        msg.content.payload.length > 0
      ) {
        console.log(msg.content.payload[0].data);
        console.log(msg.content.payload[0].data['x-vendor/papyri']);
        this._questionMarkSubmitted.emit(
          msg.content.payload[0].data['x-vendor/papyri']
        );
        console.log('QMS:', this._questionMarkSubmitted);
      }
    }
    this.stateChanged.emit(undefined);
  }

  private _findParent(
    args: Kernel.IAnyMessageArgs
  ): Kernel.IAnyMessageArgs | null {
    if (isHeader(args.msg.parent_header)) {
      return this._messages[args.msg.parent_header.msg_id] || null;
    }
    return null;
  }

  private _log: Kernel.IAnyMessageArgs[] = [];
  private _kernel: Kernel.IKernelConnection | null = null;
  private _messages: { [key: string]: Kernel.IAnyMessageArgs } = {};
  private _childLUT: { [key: string]: string[] } = {};
  private _roots: string[] = [];
  private _notebook: NotebookPanel | null = null;
  private _notebookTracker: INotebookTracker;
  private _questionMarkSubmitted = new Signal<KernelSpyModel, string>(this);
}

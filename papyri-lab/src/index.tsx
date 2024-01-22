// Entry point for the Papyri jupyter Lab extension.
//
import { KernelSpyModel } from './kernelspy';
import { PapyriPanel } from './widgets';
import {
  ILayoutRestorer,
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import {
  ICommandPalette,
  MainAreaWidget,
  WidgetTracker
} from '@jupyterlab/apputils';
import { INotebookTracker } from '@jupyterlab/notebook';
import { ISettingRegistry } from '@jupyterlab/settingregistry';

/**
 * Initialization data for the papyri-lab extension.
 */
const plugin: JupyterFrontEndPlugin<void> = {
  id: 'papyri-lab:plugin',
  description: 'A JupyterLab extension for papyri',
  autoStart: true,
  optional: [ISettingRegistry, ILayoutRestorer],
  requires: [ICommandPalette, INotebookTracker],
  activate: (
    app: JupyterFrontEnd,
    palette: ICommandPalette,
    notebookTracker: INotebookTracker,
    settingRegistry: ISettingRegistry | null,
    restorer: ILayoutRestorer | null
  ) => {
    console.log('JupyterLab extension papyri-lab is activated!');
    if (settingRegistry) {
      settingRegistry
        .load(plugin.id)
        .then(settings => {
          console.log('papyri-lab settings loaded:', settings.composite);
        })
        .catch(reason => {
          console.error('Failed to load settings for papyri-lab.', reason);
        });
    }

    const newWidget = () => {
      // Create a blank content widget inside of a MainAreaWidget
      return new MainAreaWidget({ content: new PapyriPanel() });
    };
    let widget: MainAreaWidget<PapyriPanel>;

    // Track and restore the widget state
    const tracker = new WidgetTracker<MainAreaWidget<PapyriPanel>>({
      namespace: 'papyri'
    });

    const command: string = 'papyri:open';
    app.commands.addCommand(command, {
      label: 'Open Papyri Browser',
      execute: () => {
        // Regenerate the widget if disposed
        if (!widget || widget.isDisposed) {
          widget = newWidget();
        }
        if (!tracker.has(widget)) {
          // Track the state of the widget for later restoration
          tracker.add(widget);
        }
        if (!widget.isAttached) {
          // Attach the widget to the main work area if it's not there
          app.shell.add(widget, 'main');
        }
        // Activate the widget
        app.shell.activateById(widget.id);
      }
    });

    // Add the command to the palette.
    palette.addItem({ command, category: 'Tutorial' });

    if (restorer) {
      restorer.restore(tracker, {
        command,
        name: () => 'papyri'
      });
    }

    const kernelSpy = new KernelSpyModel(notebookTracker);
    kernelSpy.questionMarkSubmitted.connect((_, args: any) => {
      console.info('KSpy questionMarkSubmitted args:', args);
      if (args !== undefined) {
        widget.content.updateSeachTerm(args.qualname);
        console.info('DO your thing here.');
      }
    });
  }
};

export default plugin;

from FeatureCloud.app.engine.app import AppState, app_state, Role, LogLevel
from federated_dca.utils import load_params, trainInstince, average_model_params
import bios

@app_state('initial', Role.BOTH)
class InitialState(AppState):

    def register(self):
        self.register_transition('train', Role.BOTH)

    def run(self):
        if self.is_coordinator:
            self.config = bios.read('/mnt/input/config.yml')['fc_dca']
            train_instince = trainInstince(self.config)
            self.log('Send initial Model to Clients')
            self.broadcast_data(train_instince.model.state_dict())
            init_model_state = self.await_data()
            self.store('train_instince', train_instince)
            return 'train'
        else:
            self.config = bios.read('/mnt/input/config.yml')['fc_dca']
            train_instince = trainInstince(self.config)
            init_model_state = self.await_data()
            self.log(f'Received initial Model state')
            train_instince.model.load_state_dict(init_model_state)
            self.store('train_instince', train_instince)
            return 'train'

@app_state('train', Role.BOTH)
class TrainState(AppState):
    def register(self):
        self.register_transition('aggregate', Role.COORDINATOR)
        self.register_transition('obtain', Role.PARTICIPANT)
        self.register_transition('terminal')
    
    def run(self):
        train_instince = self.load('train_instince')
        train_instince.train(self.update, self.log, self.id)
        model_weights = train_instince.get_weights()
        self.log(f'Send Model weights')
        self.send_data_to_coordinator(model_weights)
        if train_instince.finished_training:
            train_instince.finish()
            return 'terminal'
        elif self.is_coordinator:
            return 'aggregate'
        else:
            return 'obtain'

@app_state('aggregate', Role.COORDINATOR)
class GlobalAggregate(AppState):
    def register(self):
        self.register_transition('obtain', Role.COORDINATOR)
    
    def run(self):
        model_states = self.gather_data()
        self.log('Recived Model weights')
        model_state = average_model_params(model_states)
        self.log('Send updated Model weights')
        self.broadcast_data(model_state)
        return 'obtain'

@app_state('obtain', Role.BOTH)
class LocalUpdate(AppState):
    def register(self):
        self.register_transition('train', Role.BOTH)
    
    def run(self):
        updated_weights = self.await_data()
        self.log(f'{self.id} received updated Model weights')
        train_instince = self.load('train_instince')
        train_instince.set_weights(updated_weights)
        return 'train'

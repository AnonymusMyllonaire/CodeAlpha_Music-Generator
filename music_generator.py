# music_generator.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import pickle
import os
import glob
from collections import Counter
import matplotlib.pyplot as plt
from IPython.display import Audio
import warnings
warnings.filterwarnings('ignore')

# Try to import music21, but provide fallback if not available
try:
    from music21 import converter, instrument, note, chord, stream
    MUSIC21_AVAILABLE = True
except ImportError:
    print("music21 not available. Using simplified MIDI processing.")
    MUSIC21_AVAILABLE = False

class MusicGenerator:
    def __init__(self, sequence_length=100):
        self.notes = []
        self.sequence_length = sequence_length
        self.n_vocab = 0
        self.network_input = None
        self.network_output = None
        self.model = None
        self.note_to_int = {}
        self.int_to_note = {}
        
    def load_midi_files(self, path='midi_files/*.mid'):
        """Load MIDI files from specified path"""
        print("Loading MIDI files...")
        
        # Create directories if they don't exist
        os.makedirs('midi_files', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
        # Get all MIDI files in the directory
        midi_files = glob.glob(path)
        
        if not midi_files:
            print("No MIDI files found. Creating sample data...")
            self.create_sample_data()
            return self.notes
        
        notes = []
        
        if not MUSIC21_AVAILABLE:
            # Fallback: create synthetic data
            print("music21 not available. Creating synthetic music data...")
            return self.create_sample_data()
        
        for file in midi_files:
            try:
                print(f"Processing: {file}")
                midi = converter.parse(file)
                
                # Get all notes and chords from the MIDI file
                notes_to_parse = None
                parts = instrument.partitionByInstrument(midi)
                
                if parts:  # file has instrument parts
                    notes_to_parse = parts.parts[0].recurse()
                else:  # file has notes in a flat structure
                    notes_to_parse = midi.flat.notes
                
                for element in notes_to_parse:
                    if isinstance(element, note.Note):
                        notes.append(str(element.pitch))
                    elif isinstance(element, chord.Chord):
                        notes.append('.'.join(str(n) for n in element.normalOrder))
                        
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue
        
        # Save notes
        self.notes = notes
        
        # Save to file for later use
        with open('data/notes.pkl', 'wb') as f:
            pickle.dump(notes, f)
        
        print(f"Loaded {len(notes)} notes from {len(midi_files)} MIDI files")
        return notes
    
    def create_sample_data(self):
        """Create sample music data for training"""
        print("Creating sample music data...")
        
        # Simple musical patterns (notes and chords)
        patterns = [
            'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5',
            'C4', 'E4', 'G4',  # C major chord
            'D4', 'F4', 'A4',  # D minor chord
            'E4', 'G4', 'B4',  # E minor chord
            'F4', 'A4', 'C5',  # F major chord
            'G4', 'B4', 'D5',  # G major chord
            'A4', 'C5', 'E5',  # A minor chord
            'B4', 'D5', 'F5',  # B diminished chord
        ]
        
        # Create longer sequence by repeating patterns
        self.notes = patterns * 50  # Create 400 notes
        
        # Save sample data
        with open('data/notes.pkl', 'wb') as f:
            pickle.dump(self.notes, f)
        
        print(f"Created {len(self.notes)} sample notes")
        return self.notes
    
    def load_notes_from_file(self):
        """Load notes from saved file"""
        try:
            with open('data/notes.pkl', 'rb') as f:
                self.notes = pickle.load(f)
            print(f"Loaded {len(self.notes)} notes from file")
            return self.notes
        except FileNotFoundError:
            print("No saved notes found. Creating sample data...")
            return self.create_sample_data()
    
    def prepare_sequences(self):
        """Prepare the sequences used by the Neural Network"""
        if not self.notes:
            self.load_notes_from_file()
        
        # Get all pitch names
        pitchnames = sorted(set(self.notes))
        
        # Create a dictionary to map pitches to integers
        self.note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
        self.int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
        
        self.n_vocab = len(pitchnames)
        print(f"Vocabulary size: {self.n_vocab}")
        
        # Create input sequences and corresponding outputs
        network_input = []
        network_output = []
        
        # Create input sequences and the corresponding outputs
        for i in range(0, len(self.notes) - self.sequence_length, 1):
            sequence_in = self.notes[i:i + self.sequence_length]
            sequence_out = self.notes[i + self.sequence_length]
            
            network_input.append([self.note_to_int[char] for char in sequence_in])
            network_output.append(self.note_to_int[sequence_out])
        
        n_patterns = len(network_input)
        
        if n_patterns == 0:
            print("Not enough data to create sequences. Creating more sample data...")
            self.create_sample_data()
            return self.prepare_sequences()
        
        print(f"Total patterns: {n_patterns}")
        
        # Reshape the input into a format compatible with LSTM layers
        self.network_input = np.reshape(network_input, (n_patterns, self.sequence_length, 1))
        
        # Normalize input
        self.network_input = self.network_input / float(self.n_vocab)
        
        # One-hot encode the output
        self.network_output = tf.keras.utils.to_categorical(network_output, num_classes=self.n_vocab)
        
        return self.network_input, self.network_output
    
    def create_model(self):
        """Create the LSTM model"""
        if self.n_vocab == 0:
            self.prepare_sequences()
        
        # Use a simpler model for reliability
        model = Sequential([
            LSTM(256, input_shape=(self.sequence_length, 1), return_sequences=True),
            Dropout(0.3),
            LSTM(256),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(self.n_vocab, activation='softmax')
        ])
        
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']
        )
        
        self.model = model
        print("Model created successfully!")
        model.summary()
        return model
    
    def train_model(self, epochs=50, batch_size=64):
        """Train the model"""
        if self.model is None:
            self.create_model()
        
        if self.network_input is None:
            self.prepare_sequences()
        
        # Create weights directory
        os.makedirs('weights', exist_ok=True)
        
        # Callbacks
        filepath = "weights/weights-{epoch:02d}-{loss:.4f}.h5"
        checkpoint = ModelCheckpoint(
            filepath,
            monitor='loss',
            verbose=1,
            save_best_only=True,
            mode='min'
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
        
        callbacks_list = [checkpoint, reduce_lr]
        
        print("Starting training...")
        
        # Train the model
        history = self.model.fit(
            self.network_input, 
            self.network_output,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1,
            validation_split=0.2
        )
        
        # Plot training history
        self.plot_training_history(history)
        
        return history
    
    def plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
    
    def generate_music(self, start_sequence=None, length=200, temperature=1.0):
        """Generate new music"""
        if self.model is None:
            # Try to load the model first
            try:
                self.load_model()
            except:
                print("No trained model found. Please train the model first.")
                return None
        
        if self.n_vocab == 0:
            self.prepare_sequences()
        
        # Generate starting sequence if not provided
        if start_sequence is None:
            # Start with a random sequence from the input
            start_idx = np.random.randint(0, len(self.network_input) - 1)
            pattern = list(self.network_input[start_idx].flatten() * self.n_vocab)
        else:
            # Convert start sequence to pattern
            pattern = [self.note_to_int[note] for note in start_sequence]
        
        # Ensure pattern is the correct length
        if len(pattern) > self.sequence_length:
            pattern = pattern[-self.sequence_length:]
        elif len(pattern) < self.sequence_length:
            # Pad with zeros if too short
            pattern = [0] * (self.sequence_length - len(pattern)) + pattern
        
        prediction_output = []
        
        print("Generating music...")
        for note_index in range(length):
            # Prepare input pattern
            prediction_input = np.reshape(pattern, (1, self.sequence_length, 1))
            prediction_input = prediction_input / float(self.n_vocab)
            
            # Make prediction
            prediction = self.model.predict(prediction_input, verbose=0)
            
            # Apply temperature
            prediction = np.log(prediction + 1e-7) / temperature  # Add small epsilon to avoid log(0)
            exp_preds = np.exp(prediction)
            prediction = exp_preds / np.sum(exp_preds)
            
            # Sample from the distribution
            index = np.random.choice(range(self.n_vocab), p=prediction[0])
            
            # Convert index to note and append to output
            result = self.int_to_note[index]
            prediction_output.append(result)
            
            # Update pattern
            pattern.append(index)
            pattern = pattern[1:]
        
        print("Music generation completed!")
        return prediction_output
    
    def create_midi(self, prediction_output, filename="generated_music.mid"):
        """Convert the output from the prediction to notes and create a MIDI file"""
        if not MUSIC21_AVAILABLE:
            print("music21 not available. Saving notes to text file instead.")
            self.save_notes_to_file(prediction_output, filename)
            return None
        
        offset = 0
        output_notes = []
        
        # Create note and chord objects based on the values generated by the model
        for pattern in prediction_output:
            # Pattern is a chord
            if ('.' in pattern) and pattern.replace('.', '').isdigit():
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            # Pattern is a note
            else:
                try:
                    new_note = note.Note(pattern)
                    new_note.offset = offset
                    new_note.storedInstrument = instrument.Piano()
                    output_notes.append(new_note)
                except:
                    # If it's not a valid note, skip it
                    continue
            
            # Increase offset each iteration so that notes don't stack
            offset += 0.5
        
        # Create output directory
        os.makedirs('output', exist_ok=True)
        
        # Create a stream object
        midi_stream = stream.Stream(output_notes)
        
        # Write to MIDI file
        midi_stream.write('midi', fp=f'output/{filename}')
        print(f"MIDI file saved as: output/{filename}")
        
        return midi_stream
    
    def save_notes_to_file(self, notes, filename):
        """Save generated notes to a text file (fallback when music21 is not available)"""
        os.makedirs('output', exist_ok=True)
        txt_filename = filename.replace('.mid', '.txt')
        
        with open(f'output/{txt_filename}', 'w') as f:
            for note in notes:
                f.write(f"{note}\n")
        
        print(f"Notes saved to: output/{txt_filename}")
    
    def load_model(self, weights_path=None):
        """Load a pre-trained model"""
        if weights_path is None:
            # Find the latest weights file
            weights_files = glob.glob("weights/*.h5")
            if not weights_files:
                weights_files = glob.glob("weights/*.hdf5")
            
            if weights_files:
                # Get the most recent file
                weights_path = max(weights_files, key=os.path.getctime)
            else:
                raise FileNotFoundError("No weights files found. Please train the model first.")
        
        if self.model is None:
            self.create_model()
        
        self.model.load_weights(weights_path)
        print(f"Model weights loaded from: {weights_path}")
    
    def play_generated_music(self, prediction_output):
        """For Jupyter notebook: try to play the generated music"""
        if not MUSIC21_AVAILABLE:
            print("music21 not available for audio playback.")
            return None
        
        try:
            midi_stream = self.create_midi(prediction_output, "temp_playback.mid")
            if midi_stream:
                # For Jupyter notebook playback
                midi_stream.write('midi', fp='temp_playback.mid')
                return Audio('temp_playback.mid')
        except Exception as e:
            print(f"Could not play music: {e}")
            return None

# Simplified demo function
def demo_music_generation():
    """Demo function for music generation"""
    print("🎵 AI Music Generation Demo")
    print("=" * 40)
    
    # Initialize music generator
    generator = MusicGenerator(sequence_length=50)
    
    # Step 1: Load or create data
    print("\n1. Preparing data...")
    generator.load_notes_from_file()
    
    # Step 2: Prepare sequences
    print("\n2. Preparing sequences...")
    generator.prepare_sequences()
    
    # Step 3: Create model
    print("\n3. Creating model...")
    generator.create_model()
    
    # Step 4: Train model
    print("\n4. Training model...")
    print("Training for 20 epochs (this may take a few minutes)...")
    generator.train_model(epochs=20, batch_size=32)
    
    # Step 5: Generate music
    print("\n5. Generating music...")
    generated_notes = generator.generate_music(length=100, temperature=0.8)
    
    if generated_notes:
        # Step 6: Save results
        print("\n6. Saving results...")
        midi_stream = generator.create_midi(generated_notes, "demo_output.mid")
        
        print("\n🎉 Demo completed successfully!")
        print("Generated music saved in 'output/' directory")
        
        # Try to play the music
        try:
            audio = generator.play_generated_music(generated_notes)
            if audio:
                return audio
        except:
            print("Could not play audio in this environment")
    
    return None

# Quick test function
def quick_test():
    """Quick test without training"""
    generator = MusicGenerator(sequence_length=30)
    
    # Create sample data
    generator.create_sample_data()
    generator.prepare_sequences()
    generator.create_model()
    
    # Generate some music
    generated_notes = generator.generate_music(length=50, temperature=1.0)
    
    if generated_notes:
        print("Generated notes:", generated_notes[:10])
        generator.save_notes_to_file(generated_notes, "quick_test.txt")
        print("Quick test completed! Check output/quick_test.txt")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('midi_files', exist_ok=True)
    os.makedirs('weights', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    print("Music Generation with AI")
    print("1. Demo (train and generate)")
    print("2. Quick test (no training)")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "2":
        quick_test()
    else:
        demo_music_generation()
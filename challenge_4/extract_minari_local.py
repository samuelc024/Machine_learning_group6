"""Extrae demostraciones desde datasets Minari locales y guarda un .npz y un resumen.

Uso:
    python "group6/extract_minari_local.py" --dataset contains:venture --max-steps 50000

Si no pasas argumentos intentará localizar un dataset local que contenga 'venture'.
"""
import argparse
import os
import numpy as np


def find_local_dataset(substring='venture'):
    try:
        import minari
    except Exception:
        raise RuntimeError('minari no está instalado en este entorno')
    local = minari.list_local_datasets()
    # local dataset ids may be like 'atari/venture/expert-v0'
    candidates = [d for d in local if substring in d.lower()]
    return candidates


def load_episodes_from_minari(dataset_id):
    import minari
    ds = minari.load_dataset(dataset_id)
    # Try filter_episodes
    try:
        episodes = ds.filter_episodes(condition=lambda ep: True)
        return episodes
    except Exception:
        pass
    for attr in ('episodes', 'iter_episodes', 'get_episodes'):
        if hasattr(ds, attr):
            candidate = getattr(ds, attr)
            try:
                return candidate() if callable(candidate) else candidate
            except Exception:
                continue
    raise RuntimeError('No se pudieron extraer episodios del dataset Minari')


def process_episodes(episodes, max_steps):
    obs_list = []
    act_list = []
    returns = []
    steps_collected = 0
    def _first_non_none(*vals):
        for v in vals:
            if v is not None:
                return v
        return None

    for episode in episodes:
        # multiple attribute names possible; avoid truthiness checks on arrays
        obs = _first_non_none(getattr(episode, 'observations', None), getattr(episode, 'obs', None), getattr(episode, 'frames', None))
        acts = _first_non_none(getattr(episode, 'actions', None), getattr(episode, 'acts', None))
        rewards = _first_non_none(getattr(episode, 'rewards', None), getattr(episode, 'rews', None))

        if obs is None or acts is None:
            # maybe episode is a dict-like
            try:
                if isinstance(episode, dict):
                    obs = episode.get('observations') or episode.get('obs')
                    acts = episode.get('actions')
                    rewards = episode.get('rewards')
            except Exception:
                pass

        if obs is None or acts is None:
            continue

        obs = np.array(obs)
        acts = np.array(acts)
        rew = np.array(rewards) if rewards is not None else None

        if obs.size == 0 or acts.size == 0:
            continue

        if len(obs) > len(acts):
            obs = obs[:-1]

        take = min(len(acts), max_steps - steps_collected)
        if take <= 0:
            break

        obs_list.append(obs[:take])
        act_list.append(acts[:take])

        if rew is not None:
            returns.append(float(np.sum(rew)))
        steps_collected += take

    if obs_list:
        observations = np.concatenate(obs_list, axis=0)
        actions = np.concatenate(act_list, axis=0)
    else:
        observations = np.array([])
        actions = np.array([])
    return observations, actions, returns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None,
                        help='Minari dataset id to use. If omitted, finds a local dataset containing "venture"')
    parser.add_argument('--max-steps', type=int, default=50000)
    parser.add_argument('--out', type=str, default='demos_venture.npz')
    args = parser.parse_args()

    try:
        import minari
    except Exception:
        print('Error: minari no está instalado en este entorno. Instala minari y reintenta.')
        return

    dataset_id = args.dataset
    if dataset_id is None:
        candidates = find_local_dataset('venture')
        if not candidates:
            print('No se encontró ningún dataset local de Minari que contenga "venture". Ejecuta minari.list_remote_datasets() o descarga con minari.download_dataset().')
            return
        dataset_id = candidates[0]
        print(f"Usando dataset local: {dataset_id}")
    else:
        print(f"Usando dataset: {dataset_id}")

    try:
        episodes = load_episodes_from_minari(dataset_id)
    except Exception as e:
        print('Error cargando episodios desde Minari:', e)
        return

    observations, actions, returns = process_episodes(episodes, args.max_steps)

    if observations.size == 0 or actions.size == 0:
        print('No se extrajeron observaciones/acciones del dataset.')
        return

    np.savez_compressed(args.out, observations=observations, actions=actions)
    print(f'Archivo guardado: {args.out} (observations.shape={observations.shape}, actions.shape={actions.shape})')

    # guardar resumen
    returns_arr = np.array(returns) if returns else np.array([])
    with open('demos_info.txt', 'w', encoding='utf-8') as f:
        f.write(f'Dataset: {dataset_id}\n')
        f.write(f'Path minari local: {os.path.expanduser("~")}/.minari/datasets/\n')
        f.write(f'Episodios contados: {len(returns)}\n')
        if returns_arr.size:
            f.write(f'Rendimiento (media ± std): {returns_arr.mean():.1f} ± {returns_arr.std():.1f}\n')
            f.write(f'Rendimiento max: {returns_arr.max():.1f}\n')
        f.write(f'Observations shape: {observations.shape}\n')
        f.write(f'Actions shape: {actions.shape}\n')
    print('Resumen guardado en demos_info.txt')


if __name__ == '__main__':
    main()

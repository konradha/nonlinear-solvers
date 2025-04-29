import argparse
import os
import sys
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def format_params(attrs):
    lines = []
    lines.append("--- System & Grid ---")
    lines.append(f"Type: {attrs.get('problem_type', 'N/A')}")
    lines.append(f"Grid: {attrs.get('nx', '?')}x{attrs.get('ny', '?')}x{attrs.get('nz', '?')}")
    lines.append(f"Domain: [-{attrs.get('Lx', '?')}, {attrs.get('Lx', '?')}]^3") 
    lines.append(f"BC: {attrs.get('boundary_condition', 'N/A')}")

    lines.append("\n--- Time ---")
    lines.append(f"T = {attrs.get('T', '?')}")
    lines.append(f"Steps (nt): {attrs.get('nt', '?')}")
    lines.append(f"Snapshots: {attrs.get('num_snapshots', '?')}")
    lines.append(f"Walltime: {attrs.get('elapsed_time', -1):.2f} s")

    lines.append("\n--- Coefficients ---")
    parent_dir_name = Path(attrs.get('filename_hdf5', '')).parent.parent.name 
    
    c_m_pair_inferred = 'N/A'
    prefixes = ["kge_kink_field_", "nlse_multi_soliton_state_", "nlse_skyrmion_tube_"]
    for prefix in prefixes:
        if parent_dir_name.startswith(prefix):
            potential_pair = parent_dir_name[len(prefix):]
            if potential_pair.endswith("_with_visuals"):
                 potential_pair = potential_pair[:-len("_with_visuals")]
            c_m_pair_inferred = potential_pair
            break
            
    lines.append(f"c, m Pair Type: {c_m_pair_inferred}")
    lines.append(f"c Type (metadata): {attrs.get('anisotropy_type', 'N/A')}")
    lines.append(f"m Type (metadata): {attrs.get('focusing_type', 'N/A')}")

    lines.append("\n--- Initial Condition ---")
    lines.append(f"Phenomenon: {attrs.get('phenomenon', 'N/A')}")
    phenom_keys = sorted([k for k in attrs if k.startswith('phenomenon_')])
    for k in phenom_keys:
        key_short = k.replace('phenomenon_', '')
        val = attrs[k]
        try:
            if isinstance(val, str) and val.startswith('(') and val.endswith(')'):
                 eval_val = eval(val)
                 if isinstance(eval_val, tuple) and len(eval_val) == 2:
                     lines.append(f"{key_short}: ({eval_val[0]:.2g}, {eval_val[1]:.2g})")
                 else:
                     lines.append(f"{key_short}: {val}") 
            else:
                 lines.append(f"{key_short}: {val}")
        except:
             lines.append(f"{key_short}: {val}") 
             
    if 'phenomenon_velocity_type' in attrs:
         lines.append(f"velocity_type: {attrs['phenomenon_velocity_type']}")
        
    return "\n".join(lines)


def plot_trajectory_slices(h5_filepath, output_dir):
    filepath = Path(h5_filepath)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plot_filename = output_path / f"{filepath.stem}_slices.png"

    try:
        with h5py.File(filepath, 'r') as f:
            metadata = dict(f['metadata'].attrs)
            metadata['filename_hdf5'] = str(h5_filepath) 
            grid_attrs = dict(f['grid'].attrs)
            time_attrs = dict(f['time'].attrs)
            metadata.update(grid_attrs)
            metadata.update(time_attrs)
            
            problem_type = metadata.get('problem_type', 'unknown')
            is_kge = (problem_type == 'klein_gordon')
            
            u = f['u'][()]
            if is_kge and 'v' in f:
                v = f['v'][()] 
            else:
                v = None 
                
            c = f['anisotropy/c'][()]
            m = f['focusing/m'][()]
            
            nx = metadata['nx']
            ny = metadata['ny']
            nz = metadata['nz']
            Lx = metadata['Lx']
            Ly = metadata['Ly']
            
            T = metadata['T']
            num_snapshots = metadata['num_snapshots']
            
            slice_idx_z = nz // 2
            c_slice = c[:, :, slice_idx_z]
            m_slice = m[:, :, slice_idx_z]
            
            times_to_plot = [0, T/4, T/2, 3*T/4, T]
            snapshot_indices = np.linspace(0, num_snapshots - 1, num_snapshots, dtype=int)
            time_points = np.linspace(0, T, num_snapshots)
            
            plot_indices = []
            actual_times = []
            for t_target in times_to_plot:
                idx = np.argmin(np.abs(time_points - t_target))
                plot_indices.append(snapshot_indices[idx])
                actual_times.append(time_points[idx])
                
            u_slices = u[plot_indices, :, :, slice_idx_z]
            if v is not None:
                v_slices = v[plot_indices, :, :, slice_idx_z]
            else:
                v_slices = None
            
            if np.iscomplexobj(u_slices):
                u_mag_min = np.min(np.abs(u_slices))
                u_mag_max = np.max(np.abs(u_slices))
                u_phase_min = -np.pi
                u_phase_max = np.pi
                cmap_u_mag = 'viridis'
                cmap_u_phase = 'hsv'
            else: 
                u_min = np.min(u_slices)
                u_max = np.max(u_slices)
                if np.isclose(u_min, u_max): 
                    u_min -= 0.1
                    u_max += 0.1
                cmap_u = 'coolwarm' 
                
                if v_slices is not None:
                    v_min = np.min(v_slices)
                    v_max = np.max(v_slices)
                    if np.isclose(v_min, v_max):
                        v_min -= 0.1
                        v_max += 0.1
                    cmap_v = 'coolwarm'
            
            c_min, c_max = np.min(c), np.max(c)
            m_min, m_max = np.min(m), np.max(m)
            m_cmap = 'viridis'
            m_vmin, m_vmax = m_min, m_max
            if m_min < -1e-6 and m_max > 1e-6:
                m_abs_max = max(abs(m_min), abs(m_max))
                m_vmin, m_vmax = -m_abs_max, m_abs_max
                m_cmap = 'coolwarm' 
                
            num_rows = 5 if is_kge else 4 
            num_cols_u = 2 if not np.iscomplexobj(u_slices) else 4 
            
            fig = plt.figure(figsize=(5 * num_cols_u, 4 * num_rows + 1)) 
            gs = gridspec.GridSpec(num_rows, num_cols_u + 1, figure=fig, width_ratios=[1]*num_cols_u + [0.8]) 

            ax_text = fig.add_subplot(gs[:, -1]) 
            ax_text.axis('off')
            
            ax_c = fig.add_subplot(gs[0, 0])
            im_c = ax_c.imshow(c_slice.T, extent=[-Lx, Lx, -Ly, Ly], origin='lower', cmap='viridis', vmin=c_min, vmax=c_max, interpolation='nearest')
            ax_c.set_title('$c(x, y, z_0)$')
            ax_c.set_xlabel('x'); ax_c.set_ylabel('y'); ax_c.set_aspect('equal')
            fig.colorbar(im_c, ax=ax_c, shrink=0.8)

            ax_m = fig.add_subplot(gs[0, 1])
            im_m = ax_m.imshow(m_slice.T, extent=[-Lx, Lx, -Ly, Ly], origin='lower', cmap=m_cmap, vmin=m_vmin, vmax=m_vmax, interpolation='nearest')
            ax_m.set_title('$m(x, y, z_0)$')
            ax_m.set_xlabel('x'); ax_m.set_ylabel('y'); ax_m.set_aspect('equal')
            fig.colorbar(im_m, ax=ax_m, shrink=0.8)
            
            plot_idx = 0
            for r in range(1, num_rows): 
                for c_u in range(2): 
                    if plot_idx >= len(plot_indices): break
                    
                    t_val = actual_times[plot_idx]
                    u_slice = u_slices[plot_idx]

                    if np.iscomplexobj(u_slice):
                        col_start_u = c_u * 2 
                        ax_u_mag = fig.add_subplot(gs[r, col_start_u])
                        im_u_mag = ax_u_mag.imshow(np.abs(u_slice).T, extent=[-Lx, Lx, -Ly, Ly], origin='lower', cmap=cmap_u_mag, vmin=u_mag_min, vmax=u_mag_max, interpolation='nearest')
                        ax_u_mag.set_title(f'$|u(t={t_val:.2f})|$')
                        ax_u_mag.set_xlabel('x'); ax_u_mag.set_ylabel('y'); ax_u_mag.set_aspect('equal')
                        fig.colorbar(im_u_mag, ax=ax_u_mag, shrink=0.8)

                        ax_u_phase = fig.add_subplot(gs[r, col_start_u + 1])
                        im_u_phase = ax_u_phase.imshow(np.angle(u_slice).T, extent=[-Lx, Lx, -Ly, Ly], origin='lower', cmap=cmap_u_phase, vmin=u_phase_min, vmax=u_phase_max, interpolation='nearest')
                        ax_u_phase.set_title(f'$arg(u(t={t_val:.2f}))$')
                        ax_u_phase.set_xlabel('x'); ax_u_phase.set_ylabel('y'); ax_u_phase.set_aspect('equal')
                        fig.colorbar(im_u_phase, ax=ax_u_phase, ticks=[-np.pi, 0, np.pi], format='%.2f', shrink=0.8)
                    
                    else: 
                         col_start_u = c_u * 1 
                         if r == num_rows - 1 and is_kge and c_u == 1: 
                             if v_slices is not None:
                                v_slice = v_slices[plot_idx]
                                ax_v = fig.add_subplot(gs[r, col_start_u])
                                im_v = ax_v.imshow(v_slice.T, extent=[-Lx, Lx, -Ly, Ly], origin='lower', cmap=cmap_v, vmin=v_min, vmax=v_max, interpolation='nearest')
                                ax_v.set_title(f'$u_t(t={t_val:.2f})$')
                                ax_v.set_xlabel('x'); ax_v.set_ylabel('y'); ax_v.set_aspect('equal')
                                fig.colorbar(im_v, ax=ax_v, shrink=0.8)
                             else: 
                                 ax_empty = fig.add_subplot(gs[r, col_start_u])
                                 ax_empty.axis('off')
                                 ax_empty.set_title('$u_t$ N/A')

                         else: 
                             ax_u = fig.add_subplot(gs[r, col_start_u])
                             im_u = ax_u.imshow(u_slice.T, extent=[-Lx, Lx, -Ly, Ly], origin='lower', cmap=cmap_u, vmin=u_min, vmax=u_max, interpolation='nearest')
                             ax_u.set_title(f'$u(t={t_val:.2f})$')
                             ax_u.set_xlabel('x'); ax_u.set_ylabel('y'); ax_u.set_aspect('equal')
                             fig.colorbar(im_u, ax=ax_u, shrink=0.8)

                    plot_idx += 1
                if plot_idx >= len(plot_indices): break 

            param_text = format_params(metadata)
            ax_text.text(0.05, 0.95, param_text, transform=ax_text.transAxes, 
                         fontsize=9, verticalalignment='top', horizontalalignment='left',
                         bbox=dict(boxstyle='round,pad=0.5', fc='aliceblue', alpha=0.7))

            plt.tight_layout(rect=[0, 0, 0.85, 1]) 
            plt.suptitle(f"Run Slices: {filepath.stem}", fontsize=14, y=0.99)
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Plot saved to {plot_filename}")

    except Exception as e:
        print(f"Error processing file {filepath}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot slices from simulation HDF5 file.")
    parser.add_argument("h5_file", type=str, help="Path to the input HDF5 file.")
    parser.add_argument("output_dir", type=str, help="Directory to save the output plot.")
    
    if len(sys.argv) < 3:
         parser.print_help()
         sys.exit(1)
         
    args = parser.parse_args()
    
    if not os.path.isfile(args.h5_file):
        print(f"Error: Input file not found: {args.h5_file}", file=sys.stderr)
        sys.exit(1)
        
    plot_trajectory_slices(args.h5_file, args.output_dir)

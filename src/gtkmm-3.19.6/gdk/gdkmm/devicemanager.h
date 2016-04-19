// Generated by gmmproc 2.47.4 -- DO NOT MODIFY!
#ifndef _GDKMM_DEVICEMANAGER_H
#define _GDKMM_DEVICEMANAGER_H


#include <glibmm/ustring.h>
#include <sigc++/sigc++.h>

/* Copyright (C) 20010 The gtkmm Development Team
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 */

#include <vector>

#include <gdkmm/device.h>
#include <gdkmm/display.h>
#include <gdk/gdk.h>


#ifndef DOXYGEN_SHOULD_SKIP_THIS
typedef struct _GdkDeviceManager GdkDeviceManager;
typedef struct _GdkDeviceManagerClass GdkDeviceManagerClass;
#endif /* DOXYGEN_SHOULD_SKIP_THIS */


#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace Gdk
{ class DeviceManager_Class; } // namespace Gdk
#endif //DOXYGEN_SHOULD_SKIP_THIS

namespace Gdk
{

/** Functions for handling input devices.
 *
 * In addition to a single pointer and keyboard for user interface input,
 * GDK contains support for a variety of input devices, including graphics
 * tablets, touchscreens and multiple pointers/keyboards interacting
 * simultaneously with the user interface. Under X, the support for multiple
 * input devices is done through the XInput 2 extension,
 * which also supports additional features such as sub-pixel positioning
 * information and additional device-dependent information.
 *
 * By default, and if the platform supports it, GDK is aware of multiple
 * keyboard/pointer pairs and multitouch devices, this behavior can be
 * changed by calling gdk_disable_multidevice() before Gdk::Display::open(),
 * although there would rarely be a reason to do that. For a widget or
 * window to be dealt as multipointer aware,
 * Gdk::Window::set_support_multidevice() or
 * Gtk::Widget::set_support_multidevice() must have been called on it.
 *
 * Conceptually, in multidevice mode there are 2 device types. Virtual
 * devices (or master devices) are represented by the pointer cursors
 * and keyboard foci that are seen on the screen. Physical devices (or
 * slave devices) represent the hardware that is controlling the virtual
 * devices, and thus have no visible cursor on the screen.
 *
 * Virtual devices are always paired, so there is a keyboard device for every
 * pointer device. Associations between devices may be inspected through
 * Gdk::Device::get_associated_device().
 *
 * There may be several virtual devices, and several physical devices could
 * be controlling each of these virtual devices. Physical devices may also
 * be "floating", which means they are not attached to any virtual device.
 *
 * By default, GDK will automatically listen for events coming from all
 * master devices, setting the Gdk::Device for all events coming from input
 * devices,
 *
 * Events containing device information are GDK_MOTION_NOTIFY,
 * GDK_BUTTON_PRESS, GDK_2BUTTON_PRESS, GDK_3BUTTON_PRESS,
 * GDK_BUTTON_RELEASE, GDK_SCROLL, GDK_KEY_PRESS, GDK_KEY_RELEASE,
 * GDK_ENTER_NOTIFY, GDK_LEAVE_NOTIFY, GDK_FOCUS_CHANGE,
 * GDK_PROXIMITY_IN, GDK_PROXIMITY_OUT, GDK_DRAG_ENTER, GDK_DRAG_LEAVE,
 * GDK_DRAG_MOTION, GDK_DRAG_STATUS, GDK_DROP_START, GDK_DROP_FINISHED
 * and GDK_GRAB_BROKEN.
 *
 * Although gdk_window_set_support_multidevice() must be called on
 * \#GdkWindows in order to support additional features of multiple pointer
 * interaction, such as multiple per-device enter/leave events, the default
 * setting will emit just one enter/leave event pair for all devices on the
 * window. See Gdk::Window::set_support_multidevice() documentation for more
 * information.
 *
 * In order to listen for events coming from other than a virtual device,
 * Gdk::Window::set_device_events() must be called. Generally, this method
 * can be used to modify the event mask for any given device.
 *
 * Input devices may also provide additional information besides X/Y.
 * For example, graphics tablets may also provide pressure and X/Y tilt
 * information. This information is device-dependent, and may be
 * queried through Gdk::Devie::get_axis(). In multidevice mode, virtual
 * devices will change axes in order to always represent the physical
 * device that is routing events through it. Whenever the physical device
 * changes, the Gdk::Device::property_n_axes() property will be notified, and
 * Gdk::Device::list_axes() will return the new device axes.
 *
 * Devices may also have associated keys or
 * macro buttons. Such keys can be globally set to map into normal X
 * keyboard events. The mapping is set using Gdk::Device::set_key().
 *
 * In order to query the device hierarchy and be aware of changes in the
 * device hierarchy (such as virtual devices being created or removed, or
 * physical devices being plugged or unplugged), GDK provides
 * Gdk::DeviceManager. On X11, multidevice support is implemented through
 * XInput 2. Unless gdk_disable_multidevice() is called, the XInput 2.x
 * Gdk::DeviceManager implementation will be used as the input source. Otherwise
 * either the core or XInput 1.x implementations will be used.
 *
 * @newin{3,0}
 */

class DeviceManager : public Glib::Object
{
  
#ifndef DOXYGEN_SHOULD_SKIP_THIS

public:
  typedef DeviceManager CppObjectType;
  typedef DeviceManager_Class CppClassType;
  typedef GdkDeviceManager BaseObjectType;
  typedef GdkDeviceManagerClass BaseClassType;

  // noncopyable
  DeviceManager(const DeviceManager&) = delete;
  DeviceManager& operator=(const DeviceManager&) = delete;

private:  friend class DeviceManager_Class;
  static CppClassType devicemanager_class_;

protected:
  explicit DeviceManager(const Glib::ConstructParams& construct_params);
  explicit DeviceManager(GdkDeviceManager* castitem);

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

public:

  DeviceManager(DeviceManager&& src) noexcept;
  DeviceManager& operator=(DeviceManager&& src) noexcept;

  ~DeviceManager() noexcept override;

  /** Get the GType for this class, for use with the underlying GObject type system.
   */
  static GType get_type()      G_GNUC_CONST;

#ifndef DOXYGEN_SHOULD_SKIP_THIS


  static GType get_base_type() G_GNUC_CONST;
#endif

  ///Provides access to the underlying C GObject.
  GdkDeviceManager*       gobj()       { return reinterpret_cast<GdkDeviceManager*>(gobject_); }

  ///Provides access to the underlying C GObject.
  const GdkDeviceManager* gobj() const { return reinterpret_cast<GdkDeviceManager*>(gobject_); }

  ///Provides access to the underlying C instance. The caller is responsible for unrefing it. Use when directly setting fields in structs.
  GdkDeviceManager* gobj_copy();

private:


protected:
  DeviceManager();

public:

  
  /** Gets the Gdk::Display associated to @a device_manager.
   * 
   * @newin{3,0}
   * 
   * @return The Gdk::Display to which
   *  @a device_manager is associated to, or #<tt>nullptr</tt>. This memory is
   * owned by GDK and must not be freed or unreferenced.
   */
  Glib::RefPtr<Display> get_display();
  
  /** Gets the Gdk::Display associated to @a device_manager.
   * 
   * @newin{3,0}
   * 
   * @return The Gdk::Display to which
   *  @a device_manager is associated to, or #<tt>nullptr</tt>. This memory is
   * owned by GDK and must not be freed or unreferenced.
   */
  Glib::RefPtr<const Display> get_display() const;

 
#ifndef GDKMM_DISABLE_DEPRECATED

  /** Returns the list of devices of type @a type currently attached to
   *  @a device_manager.
   * 
   * @newin{3,0}
   * 
   * Deprecated: 3.20, use Gdk::Seat::get_pointer(), Gdk::Seat::get_keyboard()
   * and gdk_seat_list_slaves() instead.
   * 
   * @deprecated Use Gdk::Seat::get_pointer(), Gdk::Seat::get_keyboard() and Gdk::Seat::get_slaves() instead.
   * 
   * @param type Device type to get.
   * @return A list of 
   * Gdk::Devices. The returned list must be
   * freed with Glib::list_free(). The list elements are owned by
   * GTK+ and must not be freed or unreffed.
   */
  std::vector< Glib::RefPtr<Device> > list_devices(DeviceType type);
#endif // GDKMM_DISABLE_DEPRECATED


#ifndef GDKMM_DISABLE_DEPRECATED

  /** Returns the list of devices of type @a type currently attached to
   *  @a device_manager.
   * 
   * @newin{3,0}
   * 
   * Deprecated: 3.20, use Gdk::Seat::get_pointer(), Gdk::Seat::get_keyboard()
   * and gdk_seat_list_slaves() instead.
   * 
   * @deprecated Use Gdk::Seat::get_pointer(), Gdk::Seat::get_keyboard() and Gdk::Seat::get_slaves() instead.
   * 
   * @param type Device type to get.
   * @return A list of 
   * Gdk::Devices. The returned list must be
   * freed with Glib::list_free(). The list elements are owned by
   * GTK+ and must not be freed or unreffed.
   */
  std::vector< Glib::RefPtr<const Device> > list_devices(DeviceType type) const;
#endif // GDKMM_DISABLE_DEPRECATED


#ifndef GDKMM_DISABLE_DEPRECATED

  /** Returns the client pointer, that is, the master pointer that acts as the core pointer
   * for this application. In X11, window managers may change this depending on the interaction
   * pattern under the presence of several pointers.
   * 
   * You should use this function seldomly, only in code that isn’t triggered by a Gdk::Event
   * and there aren’t other means to get a meaningful Gdk::Device to operate on.
   * 
   * @newin{3,0}
   * 
   * Deprecated: 3.20.
   * 
   * @deprecated There is no replacement.
   * 
   * @return The client pointer. This memory is
   * owned by GDK and must not be freed or unreferenced.
   */
  Glib::RefPtr<Device> get_client_pointer();
#endif // GDKMM_DISABLE_DEPRECATED


#ifndef GDKMM_DISABLE_DEPRECATED

  /** Returns the client pointer, that is, the master pointer that acts as the core pointer
   * for this application. In X11, window managers may change this depending on the interaction
   * pattern under the presence of several pointers.
   * 
   * You should use this function seldomly, only in code that isn’t triggered by a Gdk::Event
   * and there aren’t other means to get a meaningful Gdk::Device to operate on.
   * 
   * @newin{3,0}
   * 
   * Deprecated: 3.20.
   * 
   * @deprecated There is no replacement.
   * 
   * @return The client pointer. This memory is
   * owned by GDK and must not be freed or unreferenced.
   */
  Glib::RefPtr<const Device> get_client_pointer() const;
#endif // GDKMM_DISABLE_DEPRECATED


  //TODO: Signals, properties.


public:

public:
  //C++ methods used to invoke GTK+ virtual functions:

protected:
  //GTK+ Virtual Functions (override these to change behaviour):

  //Default Signal Handlers::


};

} // namespace Gdk


namespace Glib
{
  /** A Glib::wrap() method for this object.
   * 
   * @param object The C instance.
   * @param take_copy False if the result should take ownership of the C instance. True if it should take a new copy or ref.
   * @result A C++ instance that wraps this C instance.
   *
   * @relates Gdk::DeviceManager
   */
  Glib::RefPtr<Gdk::DeviceManager> wrap(GdkDeviceManager* object, bool take_copy = false);
}


#endif /* _GDKMM_DEVICEMANAGER_H */


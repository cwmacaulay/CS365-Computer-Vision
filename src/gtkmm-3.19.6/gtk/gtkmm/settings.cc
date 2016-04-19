// Generated by gmmproc 2.47.4 -- DO NOT MODIFY!


#include <glibmm.h>

#include <gtkmm/settings.h>
#include <gtkmm/private/settings_p.h>


/*
 * Copyright 1998-2002 The gtkmm Development Team
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

#include <gtk/gtk.h>

namespace Gtk
{


} //namespace Gtk


namespace
{
} // anonymous namespace

// static
GType Glib::Value<Gtk::IMPreeditStyle>::value_type()
{
  return gtk_im_preedit_style_get_type();
}

// static
GType Glib::Value<Gtk::IMStatusStyle>::value_type()
{
  return gtk_im_status_style_get_type();
}


namespace Glib
{

Glib::RefPtr<Gtk::Settings> wrap(GtkSettings* object, bool take_copy)
{
  return Glib::RefPtr<Gtk::Settings>( dynamic_cast<Gtk::Settings*> (Glib::wrap_auto ((GObject*)(object), take_copy)) );
  //We use dynamic_cast<> in case of multiple inheritance.
}

} /* namespace Glib */


namespace Gtk
{


/* The *_Class implementation: */

const Glib::Class& Settings_Class::init()
{
  if(!gtype_) // create the GType if necessary
  {
    // Glib::Class has to know the class init function to clone custom types.
    class_init_func_ = &Settings_Class::class_init_function;

    // This is actually just optimized away, apparently with no harm.
    // Make sure that the parent type has been created.
    //CppClassParent::CppObjectType::get_type();

    // Create the wrapper type, with the same class/instance size as the base type.
    register_derived_type(gtk_settings_get_type());

    // Add derived versions of interfaces, if the C type implements any interfaces:

  }

  return *this;
}


void Settings_Class::class_init_function(void* g_class, void* class_data)
{
  const auto klass = static_cast<BaseClassType*>(g_class);
  CppClassParent::class_init_function(klass, class_data);


}


Glib::ObjectBase* Settings_Class::wrap_new(GObject* object)
{
  return new Settings((GtkSettings*)object);
}


/* The implementation: */

GtkSettings* Settings::gobj_copy()
{
  reference();
  return gobj();
}

Settings::Settings(const Glib::ConstructParams& construct_params)
:
  Glib::Object(construct_params)
{

}

Settings::Settings(GtkSettings* castitem)
:
  Glib::Object((GObject*)(castitem))
{}


Settings::Settings(Settings&& src) noexcept
: Glib::Object(std::move(src))
{}

Settings& Settings::operator=(Settings&& src) noexcept
{
  Glib::Object::operator=(std::move(src));
  return *this;
}

Settings::~Settings() noexcept
{}


Settings::CppClassType Settings::settings_class_; // initialize static member

GType Settings::get_type()
{
  return settings_class_.init().get_type();
}


GType Settings::get_base_type()
{
  return gtk_settings_get_type();
}


Glib::RefPtr<Settings> Settings::get_default()
{

  Glib::RefPtr<Settings> retvalue = Glib::wrap(gtk_settings_get_default());
  if(retvalue)
    retvalue->reference(); //The function does not do a ref for us
  return retvalue;
}

Glib::RefPtr<Settings> Settings::get_for_screen(const Glib::RefPtr<Gdk::Screen>& screen)
{

  Glib::RefPtr<Settings> retvalue = Glib::wrap(gtk_settings_get_for_screen(Glib::unwrap(screen)));
  if(retvalue)
    retvalue->reference(); //The function does not do a ref for us
  return retvalue;
}


Glib::PropertyProxy< int > Settings::property_gtk_double_click_time() 
{
  return Glib::PropertyProxy< int >(this, "gtk-double-click-time");
}

Glib::PropertyProxy_ReadOnly< int > Settings::property_gtk_double_click_time() const
{
  return Glib::PropertyProxy_ReadOnly< int >(this, "gtk-double-click-time");
}

Glib::PropertyProxy< int > Settings::property_gtk_double_click_distance() 
{
  return Glib::PropertyProxy< int >(this, "gtk-double-click-distance");
}

Glib::PropertyProxy_ReadOnly< int > Settings::property_gtk_double_click_distance() const
{
  return Glib::PropertyProxy_ReadOnly< int >(this, "gtk-double-click-distance");
}

#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< bool > Settings::property_gtk_cursor_blink() 
{
  return Glib::PropertyProxy< bool >(this, "gtk-cursor-blink");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< bool > Settings::property_gtk_cursor_blink() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "gtk-cursor-blink");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< int > Settings::property_gtk_cursor_blink_time() 
{
  return Glib::PropertyProxy< int >(this, "gtk-cursor-blink-time");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< int > Settings::property_gtk_cursor_blink_time() const
{
  return Glib::PropertyProxy_ReadOnly< int >(this, "gtk-cursor-blink-time");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< int > Settings::property_gtk_cursor_blink_timeout() 
{
  return Glib::PropertyProxy< int >(this, "gtk-cursor-blink-timeout");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< int > Settings::property_gtk_cursor_blink_timeout() const
{
  return Glib::PropertyProxy_ReadOnly< int >(this, "gtk-cursor-blink-timeout");
}
#endif // GTKMM_DISABLE_DEPRECATED


Glib::PropertyProxy< bool > Settings::property_gtk_split_cursor() 
{
  return Glib::PropertyProxy< bool >(this, "gtk-split-cursor");
}

Glib::PropertyProxy_ReadOnly< bool > Settings::property_gtk_split_cursor() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "gtk-split-cursor");
}

Glib::PropertyProxy< Glib::ustring > Settings::property_gtk_theme_name() 
{
  return Glib::PropertyProxy< Glib::ustring >(this, "gtk-theme-name");
}

Glib::PropertyProxy_ReadOnly< Glib::ustring > Settings::property_gtk_theme_name() const
{
  return Glib::PropertyProxy_ReadOnly< Glib::ustring >(this, "gtk-theme-name");
}

Glib::PropertyProxy< Glib::ustring > Settings::property_gtk_key_theme_name() 
{
  return Glib::PropertyProxy< Glib::ustring >(this, "gtk-key-theme-name");
}

Glib::PropertyProxy_ReadOnly< Glib::ustring > Settings::property_gtk_key_theme_name() const
{
  return Glib::PropertyProxy_ReadOnly< Glib::ustring >(this, "gtk-key-theme-name");
}

#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< Glib::ustring > Settings::property_gtk_menu_bar_accel() 
{
  return Glib::PropertyProxy< Glib::ustring >(this, "gtk-menu-bar-accel");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< Glib::ustring > Settings::property_gtk_menu_bar_accel() const
{
  return Glib::PropertyProxy_ReadOnly< Glib::ustring >(this, "gtk-menu-bar-accel");
}
#endif // GTKMM_DISABLE_DEPRECATED


Glib::PropertyProxy< int > Settings::property_gtk_dnd_drag_threshold() 
{
  return Glib::PropertyProxy< int >(this, "gtk-dnd-drag-threshold");
}

Glib::PropertyProxy_ReadOnly< int > Settings::property_gtk_dnd_drag_threshold() const
{
  return Glib::PropertyProxy_ReadOnly< int >(this, "gtk-dnd-drag-threshold");
}

Glib::PropertyProxy< Glib::ustring > Settings::property_gtk_font_name() 
{
  return Glib::PropertyProxy< Glib::ustring >(this, "gtk-font-name");
}

Glib::PropertyProxy_ReadOnly< Glib::ustring > Settings::property_gtk_font_name() const
{
  return Glib::PropertyProxy_ReadOnly< Glib::ustring >(this, "gtk-font-name");
}

Glib::PropertyProxy< Glib::ustring > Settings::property_gtk_modules() 
{
  return Glib::PropertyProxy< Glib::ustring >(this, "gtk-modules");
}

Glib::PropertyProxy_ReadOnly< Glib::ustring > Settings::property_gtk_modules() const
{
  return Glib::PropertyProxy_ReadOnly< Glib::ustring >(this, "gtk-modules");
}

Glib::PropertyProxy< int > Settings::property_gtk_xft_antialias() 
{
  return Glib::PropertyProxy< int >(this, "gtk-xft-antialias");
}

Glib::PropertyProxy_ReadOnly< int > Settings::property_gtk_xft_antialias() const
{
  return Glib::PropertyProxy_ReadOnly< int >(this, "gtk-xft-antialias");
}

Glib::PropertyProxy< int > Settings::property_gtk_xft_hinting() 
{
  return Glib::PropertyProxy< int >(this, "gtk-xft-hinting");
}

Glib::PropertyProxy_ReadOnly< int > Settings::property_gtk_xft_hinting() const
{
  return Glib::PropertyProxy_ReadOnly< int >(this, "gtk-xft-hinting");
}

Glib::PropertyProxy< Glib::ustring > Settings::property_gtk_xft_hintstyle() 
{
  return Glib::PropertyProxy< Glib::ustring >(this, "gtk-xft-hintstyle");
}

Glib::PropertyProxy_ReadOnly< Glib::ustring > Settings::property_gtk_xft_hintstyle() const
{
  return Glib::PropertyProxy_ReadOnly< Glib::ustring >(this, "gtk-xft-hintstyle");
}

Glib::PropertyProxy< Glib::ustring > Settings::property_gtk_xft_rgba() 
{
  return Glib::PropertyProxy< Glib::ustring >(this, "gtk-xft-rgba");
}

Glib::PropertyProxy_ReadOnly< Glib::ustring > Settings::property_gtk_xft_rgba() const
{
  return Glib::PropertyProxy_ReadOnly< Glib::ustring >(this, "gtk-xft-rgba");
}

Glib::PropertyProxy< int > Settings::property_gtk_xft_dpi() 
{
  return Glib::PropertyProxy< int >(this, "gtk-xft-dpi");
}

Glib::PropertyProxy_ReadOnly< int > Settings::property_gtk_xft_dpi() const
{
  return Glib::PropertyProxy_ReadOnly< int >(this, "gtk-xft-dpi");
}

Glib::PropertyProxy< Glib::ustring > Settings::property_gtk_cursor_theme_name() 
{
  return Glib::PropertyProxy< Glib::ustring >(this, "gtk-cursor-theme-name");
}

Glib::PropertyProxy_ReadOnly< Glib::ustring > Settings::property_gtk_cursor_theme_name() const
{
  return Glib::PropertyProxy_ReadOnly< Glib::ustring >(this, "gtk-cursor-theme-name");
}

Glib::PropertyProxy< int > Settings::property_gtk_cursor_theme_size() 
{
  return Glib::PropertyProxy< int >(this, "gtk-cursor-theme-size");
}

Glib::PropertyProxy_ReadOnly< int > Settings::property_gtk_cursor_theme_size() const
{
  return Glib::PropertyProxy_ReadOnly< int >(this, "gtk-cursor-theme-size");
}

Glib::PropertyProxy< bool > Settings::property_gtk_alternative_button_order() 
{
  return Glib::PropertyProxy< bool >(this, "gtk-alternative-button-order");
}

Glib::PropertyProxy_ReadOnly< bool > Settings::property_gtk_alternative_button_order() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "gtk-alternative-button-order");
}

Glib::PropertyProxy< bool > Settings::property_gtk_alternative_sort_arrows() 
{
  return Glib::PropertyProxy< bool >(this, "gtk-alternative-sort-arrows");
}

Glib::PropertyProxy_ReadOnly< bool > Settings::property_gtk_alternative_sort_arrows() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "gtk-alternative-sort-arrows");
}

#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< bool > Settings::property_gtk_show_input_method_menu() 
{
  return Glib::PropertyProxy< bool >(this, "gtk-show-input-method-menu");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< bool > Settings::property_gtk_show_input_method_menu() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "gtk-show-input-method-menu");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< bool > Settings::property_gtk_show_unicode_menu() 
{
  return Glib::PropertyProxy< bool >(this, "gtk-show-unicode-menu");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< bool > Settings::property_gtk_show_unicode_menu() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "gtk-show-unicode-menu");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< int > Settings::property_gtk_timeout_initial() 
{
  return Glib::PropertyProxy< int >(this, "gtk-timeout-initial");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< int > Settings::property_gtk_timeout_initial() const
{
  return Glib::PropertyProxy_ReadOnly< int >(this, "gtk-timeout-initial");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< int > Settings::property_gtk_timeout_repeat() 
{
  return Glib::PropertyProxy< int >(this, "gtk-timeout-repeat");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< int > Settings::property_gtk_timeout_repeat() const
{
  return Glib::PropertyProxy_ReadOnly< int >(this, "gtk-timeout-repeat");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< int > Settings::property_gtk_timeout_expand() 
{
  return Glib::PropertyProxy< int >(this, "gtk-timeout-expand");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< int > Settings::property_gtk_timeout_expand() const
{
  return Glib::PropertyProxy_ReadOnly< int >(this, "gtk-timeout-expand");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< Glib::ustring > Settings::property_gtk_color_scheme() 
{
  return Glib::PropertyProxy< Glib::ustring >(this, "gtk-color-scheme");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< Glib::ustring > Settings::property_gtk_color_scheme() const
{
  return Glib::PropertyProxy_ReadOnly< Glib::ustring >(this, "gtk-color-scheme");
}
#endif // GTKMM_DISABLE_DEPRECATED


Glib::PropertyProxy< bool > Settings::property_gtk_enable_animations() 
{
  return Glib::PropertyProxy< bool >(this, "gtk-enable-animations");
}

Glib::PropertyProxy_ReadOnly< bool > Settings::property_gtk_enable_animations() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "gtk-enable-animations");
}

#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< bool > Settings::property_gtk_touchscreen_mode() 
{
  return Glib::PropertyProxy< bool >(this, "gtk-touchscreen-mode");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< bool > Settings::property_gtk_touchscreen_mode() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "gtk-touchscreen-mode");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< int > Settings::property_gtk_tooltip_timeout() 
{
  return Glib::PropertyProxy< int >(this, "gtk-tooltip-timeout");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< int > Settings::property_gtk_tooltip_timeout() const
{
  return Glib::PropertyProxy_ReadOnly< int >(this, "gtk-tooltip-timeout");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< int > Settings::property_gtk_tooltip_browse_timeout() 
{
  return Glib::PropertyProxy< int >(this, "gtk-tooltip-browse-timeout");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< int > Settings::property_gtk_tooltip_browse_timeout() const
{
  return Glib::PropertyProxy_ReadOnly< int >(this, "gtk-tooltip-browse-timeout");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< int > Settings::property_gtk_tooltip_browse_mode_timeout() 
{
  return Glib::PropertyProxy< int >(this, "gtk-tooltip-browse-mode-timeout");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< int > Settings::property_gtk_tooltip_browse_mode_timeout() const
{
  return Glib::PropertyProxy_ReadOnly< int >(this, "gtk-tooltip-browse-mode-timeout");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< bool > Settings::property_gtk_keynav_cursor_only() 
{
  return Glib::PropertyProxy< bool >(this, "gtk-keynav-cursor-only");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< bool > Settings::property_gtk_keynav_cursor_only() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "gtk-keynav-cursor-only");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< bool > Settings::property_gtk_keynav_wrap_around() 
{
  return Glib::PropertyProxy< bool >(this, "gtk-keynav-wrap-around");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< bool > Settings::property_gtk_keynav_wrap_around() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "gtk-keynav-wrap-around");
}
#endif // GTKMM_DISABLE_DEPRECATED


Glib::PropertyProxy< bool > Settings::property_gtk_error_bell() 
{
  return Glib::PropertyProxy< bool >(this, "gtk-error-bell");
}

Glib::PropertyProxy_ReadOnly< bool > Settings::property_gtk_error_bell() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "gtk-error-bell");
}

#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< Gdk::Color > Settings::property_color_hash() const
{
  return Glib::PropertyProxy_ReadOnly< Gdk::Color >(this, "color-hash");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< Glib::ustring > Settings::property_gtk_file_chooser_backend() 
{
  return Glib::PropertyProxy< Glib::ustring >(this, "gtk-file-chooser-backend");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< Glib::ustring > Settings::property_gtk_file_chooser_backend() const
{
  return Glib::PropertyProxy_ReadOnly< Glib::ustring >(this, "gtk-file-chooser-backend");
}
#endif // GTKMM_DISABLE_DEPRECATED


Glib::PropertyProxy< Glib::ustring > Settings::property_gtk_print_backends() 
{
  return Glib::PropertyProxy< Glib::ustring >(this, "gtk-print-backends");
}

Glib::PropertyProxy_ReadOnly< Glib::ustring > Settings::property_gtk_print_backends() const
{
  return Glib::PropertyProxy_ReadOnly< Glib::ustring >(this, "gtk-print-backends");
}

Glib::PropertyProxy< Glib::ustring > Settings::property_gtk_print_preview_command() 
{
  return Glib::PropertyProxy< Glib::ustring >(this, "gtk-print-preview-command");
}

Glib::PropertyProxy_ReadOnly< Glib::ustring > Settings::property_gtk_print_preview_command() const
{
  return Glib::PropertyProxy_ReadOnly< Glib::ustring >(this, "gtk-print-preview-command");
}

#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< bool > Settings::property_gtk_enable_mnemonics() 
{
  return Glib::PropertyProxy< bool >(this, "gtk-enable-mnemonics");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< bool > Settings::property_gtk_enable_mnemonics() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "gtk-enable-mnemonics");
}
#endif // GTKMM_DISABLE_DEPRECATED


Glib::PropertyProxy< bool > Settings::property_gtk_enable_accels() 
{
  return Glib::PropertyProxy< bool >(this, "gtk-enable-accels");
}

Glib::PropertyProxy_ReadOnly< bool > Settings::property_gtk_enable_accels() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "gtk-enable-accels");
}

#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< int > Settings::property_gtk_recent_files_limit() 
{
  return Glib::PropertyProxy< int >(this, "gtk-recent-files-limit");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< int > Settings::property_gtk_recent_files_limit() const
{
  return Glib::PropertyProxy_ReadOnly< int >(this, "gtk-recent-files-limit");
}
#endif // GTKMM_DISABLE_DEPRECATED


Glib::PropertyProxy< Glib::ustring > Settings::property_gtk_im_module() 
{
  return Glib::PropertyProxy< Glib::ustring >(this, "gtk-im-module");
}

Glib::PropertyProxy_ReadOnly< Glib::ustring > Settings::property_gtk_im_module() const
{
  return Glib::PropertyProxy_ReadOnly< Glib::ustring >(this, "gtk-im-module");
}

Glib::PropertyProxy< int > Settings::property_gtk_recent_files_max_age() 
{
  return Glib::PropertyProxy< int >(this, "gtk-recent-files-max-age");
}

Glib::PropertyProxy_ReadOnly< int > Settings::property_gtk_recent_files_max_age() const
{
  return Glib::PropertyProxy_ReadOnly< int >(this, "gtk-recent-files-max-age");
}

Glib::PropertyProxy< int > Settings::property_gtk_fontconfig_timestamp() 
{
  return Glib::PropertyProxy< int >(this, "gtk-fontconfig-timestamp");
}

Glib::PropertyProxy_ReadOnly< int > Settings::property_gtk_fontconfig_timestamp() const
{
  return Glib::PropertyProxy_ReadOnly< int >(this, "gtk-fontconfig-timestamp");
}

Glib::PropertyProxy< Glib::ustring > Settings::property_gtk_sound_theme_name() 
{
  return Glib::PropertyProxy< Glib::ustring >(this, "gtk-sound-theme-name");
}

Glib::PropertyProxy_ReadOnly< Glib::ustring > Settings::property_gtk_sound_theme_name() const
{
  return Glib::PropertyProxy_ReadOnly< Glib::ustring >(this, "gtk-sound-theme-name");
}

Glib::PropertyProxy< bool > Settings::property_gtk_enable_input_feedback_sounds() 
{
  return Glib::PropertyProxy< bool >(this, "gtk-enable-input-feedback-sounds");
}

Glib::PropertyProxy_ReadOnly< bool > Settings::property_gtk_enable_input_feedback_sounds() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "gtk-enable-input-feedback-sounds");
}

Glib::PropertyProxy< bool > Settings::property_gtk_enable_event_sounds() 
{
  return Glib::PropertyProxy< bool >(this, "gtk-enable-event-sounds");
}

Glib::PropertyProxy_ReadOnly< bool > Settings::property_gtk_enable_event_sounds() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "gtk-enable-event-sounds");
}

#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< bool > Settings::property_gtk_enable_tooltips() 
{
  return Glib::PropertyProxy< bool >(this, "gtk-enable-tooltips");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< bool > Settings::property_gtk_enable_tooltips() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "gtk-enable-tooltips");
}
#endif // GTKMM_DISABLE_DEPRECATED


Glib::PropertyProxy< bool > Settings::property_gtk_application_prefer_dark_theme() 
{
  return Glib::PropertyProxy< bool >(this, "gtk-application-prefer-dark-theme");
}

Glib::PropertyProxy_ReadOnly< bool > Settings::property_gtk_application_prefer_dark_theme() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "gtk-application-prefer-dark-theme");
}

#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< bool > Settings::property_gtk_auto_mnemonics() 
{
  return Glib::PropertyProxy< bool >(this, "gtk-auto-mnemonics");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< bool > Settings::property_gtk_auto_mnemonics() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "gtk-auto-mnemonics");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< Gtk::PolicyType > Settings::property_gtk_visible_focus() 
{
  return Glib::PropertyProxy< Gtk::PolicyType >(this, "gtk-visible-focus");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< Gtk::PolicyType > Settings::property_gtk_visible_focus() const
{
  return Glib::PropertyProxy_ReadOnly< Gtk::PolicyType >(this, "gtk-visible-focus");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< IMPreeditStyle > Settings::property_gtk_im_preedit_style() 
{
  return Glib::PropertyProxy< IMPreeditStyle >(this, "gtk-im-preedit-style");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< IMPreeditStyle > Settings::property_gtk_im_preedit_style() const
{
  return Glib::PropertyProxy_ReadOnly< IMPreeditStyle >(this, "gtk-im-preedit-style");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< IMStatusStyle > Settings::property_gtk_im_status_style() 
{
  return Glib::PropertyProxy< IMStatusStyle >(this, "gtk-im-status-style");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< IMStatusStyle > Settings::property_gtk_im_status_style() const
{
  return Glib::PropertyProxy_ReadOnly< IMStatusStyle >(this, "gtk-im-status-style");
}
#endif // GTKMM_DISABLE_DEPRECATED


Glib::PropertyProxy< bool > Settings::property_gtk_shell_shows_app_menu() 
{
  return Glib::PropertyProxy< bool >(this, "gtk-shell-shows-app-menu");
}

Glib::PropertyProxy_ReadOnly< bool > Settings::property_gtk_shell_shows_app_menu() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "gtk-shell-shows-app-menu");
}

Glib::PropertyProxy< bool > Settings::property_gtk_shell_shows_menubar() 
{
  return Glib::PropertyProxy< bool >(this, "gtk-shell-shows-menubar");
}

Glib::PropertyProxy_ReadOnly< bool > Settings::property_gtk_shell_shows_menubar() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "gtk-shell-shows-menubar");
}

Glib::PropertyProxy< bool > Settings::property_gtk_shell_shows_desktop() 
{
  return Glib::PropertyProxy< bool >(this, "gtk-shell-shows-desktop");
}

Glib::PropertyProxy_ReadOnly< bool > Settings::property_gtk_shell_shows_desktop() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "gtk-shell-shows-desktop");
}

Glib::PropertyProxy< bool > Settings::property_gtk_enable_primary_paste() 
{
  return Glib::PropertyProxy< bool >(this, "gtk-enable-primary-paste");
}

Glib::PropertyProxy_ReadOnly< bool > Settings::property_gtk_enable_primary_paste() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "gtk-enable-primary-paste");
}

#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< int > Settings::property_gtk_menu_popup_delay() 
{
  return Glib::PropertyProxy< int >(this, "gtk-menu-popup-delay");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< int > Settings::property_gtk_menu_popup_delay() const
{
  return Glib::PropertyProxy_ReadOnly< int >(this, "gtk-menu-popup-delay");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< int > Settings::property_gtk_menu_popdown_delay() 
{
  return Glib::PropertyProxy< int >(this, "gtk-menu-popdown-delay");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< int > Settings::property_gtk_menu_popdown_delay() const
{
  return Glib::PropertyProxy_ReadOnly< int >(this, "gtk-menu-popdown-delay");
}
#endif // GTKMM_DISABLE_DEPRECATED


Glib::PropertyProxy< bool > Settings::property_gtk_label_select_on_focus() 
{
  return Glib::PropertyProxy< bool >(this, "gtk-label-select-on-focus");
}

Glib::PropertyProxy_ReadOnly< bool > Settings::property_gtk_label_select_on_focus() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "gtk-label-select-on-focus");
}

#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< guint > Settings::property_gtk_entry_password_hint_timeout() 
{
  return Glib::PropertyProxy< guint >(this, "gtk-entry-password-hint-timeout");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< guint > Settings::property_gtk_entry_password_hint_timeout() const
{
  return Glib::PropertyProxy_ReadOnly< guint >(this, "gtk-entry-password-hint-timeout");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< bool > Settings::property_gtk_menu_images() 
{
  return Glib::PropertyProxy< bool >(this, "gtk-menu-images");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< bool > Settings::property_gtk_menu_images() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "gtk-menu-images");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< int > Settings::property_gtk_menu_bar_popup_delay() 
{
  return Glib::PropertyProxy< int >(this, "gtk-menu-bar-popup-delay");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< int > Settings::property_gtk_menu_bar_popup_delay() const
{
  return Glib::PropertyProxy_ReadOnly< int >(this, "gtk-menu-bar-popup-delay");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< GtkCornerType > Settings::property_gtk_scrolled_window_placement() 
{
  return Glib::PropertyProxy< GtkCornerType >(this, "gtk-scrolled-window-placement");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< GtkCornerType > Settings::property_gtk_scrolled_window_placement() const
{
  return Glib::PropertyProxy_ReadOnly< GtkCornerType >(this, "gtk-scrolled-window-placement");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< ToolbarStyle > Settings::property_gtk_toolbar_style() 
{
  return Glib::PropertyProxy< ToolbarStyle >(this, "gtk-toolbar-style");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< ToolbarStyle > Settings::property_gtk_toolbar_style() const
{
  return Glib::PropertyProxy_ReadOnly< ToolbarStyle >(this, "gtk-toolbar-style");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< IconSize > Settings::property_gtk_toolbar_icon_size() 
{
  return Glib::PropertyProxy< IconSize >(this, "gtk-toolbar-icon-size");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< IconSize > Settings::property_gtk_toolbar_icon_size() const
{
  return Glib::PropertyProxy_ReadOnly< IconSize >(this, "gtk-toolbar-icon-size");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< bool > Settings::property_gtk_can_change_accels() 
{
  return Glib::PropertyProxy< bool >(this, "gtk-can-change-accels");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< bool > Settings::property_gtk_can_change_accels() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "gtk-can-change-accels");
}
#endif // GTKMM_DISABLE_DEPRECATED


Glib::PropertyProxy< bool > Settings::property_gtk_entry_select_on_focus() 
{
  return Glib::PropertyProxy< bool >(this, "gtk-entry-select-on-focus");
}

Glib::PropertyProxy_ReadOnly< bool > Settings::property_gtk_entry_select_on_focus() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "gtk-entry-select-on-focus");
}

#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< Glib::ustring > Settings::property_gtk_color_palette() 
{
  return Glib::PropertyProxy< Glib::ustring >(this, "gtk-color-palette");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< Glib::ustring > Settings::property_gtk_color_palette() const
{
  return Glib::PropertyProxy_ReadOnly< Glib::ustring >(this, "gtk-color-palette");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< bool > Settings::property_gtk_button_images() 
{
  return Glib::PropertyProxy< bool >(this, "gtk-button-images");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< bool > Settings::property_gtk_button_images() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "gtk-button-images");
}
#endif // GTKMM_DISABLE_DEPRECATED


Glib::PropertyProxy< Glib::ustring > Settings::property_gtk_icon_theme_name() 
{
  return Glib::PropertyProxy< Glib::ustring >(this, "gtk-icon-theme-name");
}

Glib::PropertyProxy_ReadOnly< Glib::ustring > Settings::property_gtk_icon_theme_name() const
{
  return Glib::PropertyProxy_ReadOnly< Glib::ustring >(this, "gtk-icon-theme-name");
}

#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< Glib::ustring > Settings::property_gtk_fallback_icon_theme() 
{
  return Glib::PropertyProxy< Glib::ustring >(this, "gtk-fallback-icon-theme");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< Glib::ustring > Settings::property_gtk_fallback_icon_theme() const
{
  return Glib::PropertyProxy_ReadOnly< Glib::ustring >(this, "gtk-fallback-icon-theme");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy< Glib::ustring > Settings::property_gtk_icon_sizes() 
{
  return Glib::PropertyProxy< Glib::ustring >(this, "gtk-icon-sizes");
}
#endif // GTKMM_DISABLE_DEPRECATED


#ifndef GTKMM_DISABLE_DEPRECATED

Glib::PropertyProxy_ReadOnly< Glib::ustring > Settings::property_gtk_icon_sizes() const
{
  return Glib::PropertyProxy_ReadOnly< Glib::ustring >(this, "gtk-icon-sizes");
}
#endif // GTKMM_DISABLE_DEPRECATED


Glib::PropertyProxy< bool > Settings::property_gtk_recent_files_enabled() 
{
  return Glib::PropertyProxy< bool >(this, "gtk-recent-files-enabled");
}

Glib::PropertyProxy_ReadOnly< bool > Settings::property_gtk_recent_files_enabled() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "gtk-recent-files-enabled");
}

Glib::PropertyProxy< bool > Settings::property_gtk_primary_button_warps_slider() 
{
  return Glib::PropertyProxy< bool >(this, "gtk-primary-button-warps-slider");
}

Glib::PropertyProxy_ReadOnly< bool > Settings::property_gtk_primary_button_warps_slider() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "gtk-primary-button-warps-slider");
}

Glib::PropertyProxy< Glib::ustring > Settings::property_gtk_decoration_layout() 
{
  return Glib::PropertyProxy< Glib::ustring >(this, "gtk-decoration-layout");
}

Glib::PropertyProxy_ReadOnly< Glib::ustring > Settings::property_gtk_decoration_layout() const
{
  return Glib::PropertyProxy_ReadOnly< Glib::ustring >(this, "gtk-decoration-layout");
}

Glib::PropertyProxy< bool > Settings::property_gtk_dialogs_use_header() 
{
  return Glib::PropertyProxy< bool >(this, "gtk-dialogs-use-header");
}

Glib::PropertyProxy_ReadOnly< bool > Settings::property_gtk_dialogs_use_header() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "gtk-dialogs-use-header");
}

Glib::PropertyProxy< guint > Settings::property_gtk_long_press_time() 
{
  return Glib::PropertyProxy< guint >(this, "gtk-long-press-time");
}

Glib::PropertyProxy_ReadOnly< guint > Settings::property_gtk_long_press_time() const
{
  return Glib::PropertyProxy_ReadOnly< guint >(this, "gtk-long-press-time");
}


} // namespace Gtk

